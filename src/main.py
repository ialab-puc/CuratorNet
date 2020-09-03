import numpy as np
import pandas as pd
import random
import os
import time
import json
from math import ceil
import tensorflow as tf
from Networks import ContentBasedLearn2RankNetwork_Train, TrainLogger
from sklearn.preprocessing import StandardScaler
from utils import load_embeddings_and_ids, concatenate_featmats, User, HybridScorer, VisualSimilarityHandler, VisualSimilarityHandler_ContentAndStyle, get_decaying_learning_rates


# use a single GPU because we want to be nice with other people :)
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

## PATHS
RESNET_PATH = './data/ResNet50/'
RESNEXT_PATH = './data/resnext101_32x8d_wsl/'
CLUSTER_PATH = './data/Clustering/artworkId2clusterId(resnet50+resnext101).json'
PCA_PATH = './data/PCA200/'
SALES_PATH = './data/valid_sales.csv'
ARTWORK_PATH = './data/valid_artworks.csv'
MODEL_PATH = ('./data/new_model')

# ###  Load pre-trained image embeddings
resnet50 = load_embeddings_and_ids(RESNET_PATH, 'flatten_1.npy', 'ids')
resnext101 = load_embeddings_and_ids(RESNEXT_PATH, 'features.npy', 'ids.npy')

# ###  Concatenate embeddings + z-score normalization
embedding_list = [resnet50, resnext101]

artwork_ids_set = set()
for embedding in embedding_list:
    if len(artwork_ids_set) == 0:        
        artwork_ids_set.update(embedding['index2id'])
    else:
        artwork_ids_set.intersection_update(embedding['index2id'])
artwork_ids = list(artwork_ids_set)
artwork_id2index = {_id:i for i,_id in enumerate(artwork_ids)}
n_artworks = len(artwork_ids)

featmat_list = [tmp['featmat'] for tmp in embedding_list]
id2index_list = [tmp['id2index'] for tmp in embedding_list]
concat_featmat = concatenate_featmats(artwork_ids, featmat_list, id2index_list)

concat_featmat = StandardScaler().fit_transform(concat_featmat)


# Load clusters

def load_clusters(json_path):
    with open(json_path) as f:
        artId2clustId = json.load(f)
    cluster_ids = np.full((n_artworks,), -1, dtype=int)
    for k, v in artId2clustId.items():
        cluster_ids[artwork_id2index[int(k)]] = v
    return cluster_ids, artId2clustId


def get_art_indexes_per_cluster(cluster_ids, n_clusters):
    clusterId2artworkIndexes = [[] for _ in range(n_clusters)]
    for i, cluster_id in enumerate(cluster_ids):
        clusterId2artworkIndexes[cluster_id].append(i)
    return clusterId2artworkIndexes


cluster_ids, artId2clustId = load_clusters(CLUSTER_PATH)
n_clusters = len(set(cluster_ids))
clustId2artIndexes = get_art_indexes_per_cluster(cluster_ids, n_clusters)


#  Load PCA200 embeddings
pca200 = load_embeddings_and_ids(PCA_PATH, 'embeddings.npy', 'ids.npy')


pca200_embeddings = pca200['featmat']
pca200_index2id = pca200['index2id']
pca200_id2index = pca200['id2index']

assert np.array_equal(artwork_ids, pca200_index2id)

#  Load transactions
sales_df = pd.read_csv(SALES_PATH)
artworks_df = pd.read_csv(ARTWORK_PATH)

artist_ids = np.full((n_artworks,), -1, dtype=int)
for _artworkId, _artistId in zip(artworks_df.id, artworks_df.artist_id):
    i = artwork_id2index[_artworkId]
    artist_ids[i] = _artistId

artistId2artworkIndexes = dict()
for i, _artistId in enumerate(artist_ids):
    if _artistId == -1:
        continue
    try:
        artistId2artworkIndexes[_artistId].append(i)
    except KeyError:
        artistId2artworkIndexes[_artistId] = [i]


# Collect transactions per user (making sure we hide the last nonfirst purchase basket per user)

# create list of users
user_ids = sales_df.customer_id.unique()
user_id2index = { _id:i for i,_id in enumerate(user_ids) }
users = [User(uid) for uid in user_ids]
n_users = len(user_ids)


# collect and sanity check transactions per user
sorted_sales_df = sales_df.sort_values('order_date')

# clear structures to prevent possible duplicate elements
for user in users:
    user.clear()

# collect transactions per user sorted by timestamp
for uid, aid, t in zip(sorted_sales_df.customer_id,
                       sorted_sales_df.artwork_id,
                       sorted_sales_df.order_date):
    users[user_id2index[uid]].append_transaction(
        aid, t, artwork_id2index, artist_ids, cluster_ids, cluster_ids)
    assert users[user_id2index[uid]]._uid == uid
    
# bin transctions with same timestamps into purchase baskets
for user in users:
    user.build_purchase_baskets()
    user.sanity_check_purchase_baskets()
    user.remove_last_nonfirst_purchase_basket(
        artwork_id2index, artist_ids, cluster_ids, cluster_ids)
    user.sanity_check_purchase_baskets()
    user.refresh_nonpurchased_cluster_ids(n_clusters, n_clusters)
    user.refresh_cluster_ids()
    user.refresh_artist_ids()


#  Generate training data
def hash_triple(profile, pi, ni):
    _MOD = 402653189
    _BASE = 92821
    h = 0
    for x in profile:
        h = ((h * _BASE) % _MOD + x) % _MOD
    h = ((h * _BASE) % _MOD + pi) % _MOD
    h = ((h * _BASE) % _MOD + ni) % _MOD
    return h


def sanity_check_instance(instance, pos_in_profile=True, profile_set=None):
    profile, pi, ni, ui = instance
    try:
        assert 0 <= pi < n_artworks
        assert 0 <= ni < n_artworks
        assert pi != ni        
        assert not vissimhandler.same(pi,ni)
        if ui == -1: return
        
        assert 0 <= ui < n_users
        user = users[ui]
        assert all(i in user.artwork_idxs_set for i in profile)
        user_profile = user.artwork_idxs_set if profile_set is None else profile_set
        assert ni not in user_profile
        if pos_in_profile is not None:
            assert (pi in user_profile) == pos_in_profile
        spi = hybrid_scorer.get_score(ui, user.artwork_idxs, pi)
        sni = hybrid_scorer.get_score(ui, user.artwork_idxs, ni)
        assert spi > sni

    except AssertionError:
        print('profile = ', profile)
        print('pi = ', pi)
        print('ni = ', ni)
        print('ui = ', ui)
        raise


def append_instance(container, instance, **kwargs):
    global _hash_collisions
    profile, pi, ni, ui = instance
    
    h = hash_triple(profile, pi, ni)
    if h in used_hashes:
        _hash_collisions += 1
        return False
    
    if vissimhandler.same(pi, ni):
        return False
    
    sanity_check_instance(instance, **kwargs)
    container.append(instance)
    used_hashes.add(h)
    return True


def print_triple(t):
    profile, pi, ni, ui = t
    print ('profile = ', [artwork_ids[i] for i in profile])
    print ('pi = ', artwork_ids[pi])
    print ('ni = ', artwork_ids[ni])
    print ('ui = ', user_ids[ui] if ui != -1 else -1)


def print_num_samples(sampler_func):
    def wrapper(instances_container, n_samples):        
        while True:
            len_before = len(instances_container)
            sampler_func(instances_container, n_samples)
            actual_samples = len(instances_container) - len_before
            delta = n_samples - actual_samples
            print('  target samples: %d' % n_samples)
            print('  actual samples: %d' % actual_samples)
            print('  delta: %d' % (delta))
            if delta <= 0: break
            print('  ** delta > 0 -> sampling more instances again ...')
            n_samples = delta
    return wrapper


FINE_GRAINED_THRESHOLD = 0.7
ARTIST_BOOST = 0.2
CONFIDENCE_MARGIN = 0.18

vissimhandler = VisualSimilarityHandler(cluster_ids, pca200_embeddings)

hybrid_scorer = HybridScorer(vissimhandler, artist_ids, artist_boost=ARTIST_BOOST)

vissimhandler.count = 0
used_hashes = set()
_hash_collisions = 0
train_instances = []
test_instances = []

N_STRATEGIES_FAKE = 2
N_STRATEGIES_REAL = 2
FAKE_COEF = 0.
TOTAL_SAMPLES__TRAIN = 10000000
TOTAL_SAMPLES__TEST =  TOTAL_SAMPLES__TRAIN * 0.05

N_SAMPLES_PER_FAKE_STRATEGY__TRAIN = ceil(TOTAL_SAMPLES__TRAIN * FAKE_COEF / N_STRATEGIES_FAKE)
N_SAMPLES_PER_FAKE_STRATEGY__TEST = ceil(TOTAL_SAMPLES__TEST * FAKE_COEF / N_STRATEGIES_FAKE)
N_SAMPLES_PER_REAL_STRATEGY__TRAIN = ceil(TOTAL_SAMPLES__TRAIN * (1. - FAKE_COEF) / N_STRATEGIES_REAL)
N_SAMPLES_PER_REAL_STRATEGY__TEST = ceil(TOTAL_SAMPLES__TEST * (1. - FAKE_COEF) / N_STRATEGIES_REAL)

print(N_SAMPLES_PER_FAKE_STRATEGY__TRAIN, N_SAMPLES_PER_FAKE_STRATEGY__TEST)
print(N_SAMPLES_PER_REAL_STRATEGY__TRAIN, N_SAMPLES_PER_REAL_STRATEGY__TEST)


# Sampling Triplets for Ranking

# 1) given profile, recommend profile (real users)
# Given a user's profile, all items in the profile should be ranked higher than items outside the profile (as long as the hybrid scorer agrees)

def sample_artwork_index(i):
    if random.random() <= FINE_GRAINED_THRESHOLD:
        if artist_ids[i] == -1 or random.random() <= 0.5:
            j = random.choice(clustId2artIndexes[cluster_ids[i]])
        else:
            j = random.choice(artistId2artworkIndexes[artist_ids[i]])
    else:
        c = random.randint(0, n_clusters-1)
        j = random.choice(clustId2artIndexes[c])
    return j


def sample_artwork_index__outsideprofile(profile_set, pi):
    while True:
        ni = sample_artwork_index(pi)
        if ni not in profile_set:
            return ni

@print_num_samples
def generate_samples__rank_profile_above_nonprofile(instances_container, n_samples):
    n_samples_per_user = ceil(n_samples / n_users)    
    for ui, user in enumerate(users):
        profile = user.artwork_idxs
        profile_set = user.artwork_idxs_set        
        n = n_samples_per_user
        while n > 0:
            pi = random.choice(profile)
            ni = sample_artwork_index__outsideprofile(profile_set, pi)
            spi = hybrid_scorer.get_score(ui, profile, pi)
            sni = hybrid_scorer.get_score(ui, profile, ni)
            if spi <= sni: continue
            if append_instance(instances_container, (profile, pi, ni, ui),
                               pos_in_profile=True,
                               profile_set=profile_set):
                    n -= 1

# 2) Given profile, recommend profile (fake 1-item profiles)
# Given a fake profile of a single item, such item should be ranked higher than any other item

def sample_artwork_index__nonidentical(i):
    while True:
        j = sample_artwork_index(i)
        if j != i: return j

@print_num_samples
def generate_samples__rank_single_item_above_anything_else(instances_container, n_samples):
    n_samples_per_item = ceil(n_samples / n_artworks)
    for pi in range(n_artworks):
        profile = (pi,)
        n = n_samples_per_item
        while n > 0:
            ni = sample_artwork_index__nonidentical(pi)
            if append_instance(instances_container, (profile, pi, ni, -1)):
                n -= 1

# 3) Recommend outside profile items according to hybrid recommender (real users)
# Given a user and two items outside the user's profile, the two items should be ranked according to the hybrid recommender (if a certain margin of confidence is met)

def sample_artwork_index__outside_profile(
        artists_list, clusters_list, profile_set):
    while True:
        if random.random() <= FINE_GRAINED_THRESHOLD:
            if random.random() <= 0.5:
                a = random.choice(artists_list)
                i = random.choice(artistId2artworkIndexes[a])
            else:
                c = random.choice(clusters_list)
                i = random.choice(clustId2artIndexes[c])
        else:
            c = random.randint(0, n_clusters-1)
            i = random.choice(clustId2artIndexes[c])
        if i not in profile_set: return i


@print_num_samples
def generate_samples__outside_profile__real_users(
        instances_container, n_samples):
    
    n_samples_per_user = ceil(n_samples / n_users)
    debug = 0
    for ui, user in enumerate(users):        
        profile = user.artwork_idxs
        profile_set = user.artwork_idxs_set
        artists_list = user.artist_ids
        clusters_list = user.content_cluster_ids
        n = n_samples_per_user
        user_margin = CONFIDENCE_MARGIN / len(profile)
        while n > 0:
            pi = sample_artwork_index__outside_profile(artists_list, clusters_list, profile_set)
            ni = sample_artwork_index__outside_profile(artists_list, clusters_list, profile_set)
            if pi == ni: continue
            pi_score = hybrid_scorer.get_score(ui, profile, pi)
            ni_score = hybrid_scorer.get_score(ui, profile, ni)
            if pi_score < ni_score:
                pi_score, ni_score = ni_score, pi_score
                pi, ni = ni, pi
            if pi_score < ni_score + user_margin: continue
            if append_instance(instances_container, (profile, pi, ni, ui),
                               profile_set=profile_set, pos_in_profile=False):                
                n -= 1
                if n == 0 or debug % 1000 == 0:
                    print('debug: user %d/%d : n=%d' % (ui, len(users), n), flush=True, end='\r')
                debug += 1


print('=======================================\nsampling train instances ...')
generate_samples__outside_profile__real_users(
    train_instances, n_samples=N_SAMPLES_PER_REAL_STRATEGY__TRAIN)

print('=======================================\nsampling test instances ...')
generate_samples__outside_profile__real_users(
    test_instances, n_samples=N_SAMPLES_PER_REAL_STRATEGY__TEST)

print(len(train_instances), len(test_instances))
print('hash_collisions = ', _hash_collisions)
print('visual_collisions = ', vissimhandler.count)


# 4) Recommend outside profile items according to hybrid recommender (fake 1-item profiles)
# Given a fake profile of a single item, two other items should be ranked according to the hybrid recommender (provided that a certain margin of confidence is met)

@print_num_samples
def generate_samples__outside_profile__fake_users(
        instances_container, n_samples):
    
    n_samples_per_item = ceil(n_samples / n_artworks)
    for i in range(n_artworks):
        n = n_samples_per_item
        profile = (i,)
        while n > 0:
            pi = sample_artwork_index__nonidentical(i)
            ni = sample_artwork_index__nonidentical(i)
            pi_score = hybrid_scorer.simfunc(i, pi)
            ni_score = hybrid_scorer.simfunc(i, ni)            
            if pi_score < ni_score:
                pi_score, ni_score = ni_score, pi_score
                pi, ni = ni, pi
            if pi_score < ni_score + CONFIDENCE_MARGIN: continue
            if append_instance(instances_container, (profile, pi, ni, -1)):
                n -= 1

# sort train and test instances by profile size

random.shuffle(train_instances)
train_instances.sort(key=lambda x: len(x[0]))
test_instances.sort(key=lambda x: len(x[0]))

# Train Model
def generate_minibatches(tuples, max_users_items_per_batch):
    ui_count = 0
    offset = 0
    
    batch_ranges = []
    for i, t in enumerate(tuples):
        ui_count += len(t[0]) + 3
        if ui_count > max_users_items_per_batch:
            batch_ranges.append((offset, i))
            ui_count = len(t[0]) + 3
            offset = i
            assert ui_count <= max_users_items_per_batch
    assert offset < len(tuples)
    batch_ranges.append((offset, len(tuples)))
            
    n_tuples = len(tuples)
    n_batches = len(batch_ranges)
    print('n_tuples = ', n_tuples)
    print('n_batches = ', n_batches)
    
    assert batch_ranges[0][0] == 0
    assert all(batch_ranges[i][1] == batch_ranges[i+1][0] for i in range(n_batches-1))
    assert batch_ranges[-1][1] == n_tuples
    assert sum(b[1] - b[0] for b in batch_ranges) == n_tuples
    
    profile_indexes_batches = [None] * n_batches
    profile_size_batches = [None] * n_batches
    positive_index_batches = [None] * n_batches
    negative_index_batches = [None] * n_batches
    
    for i, (jmin, jmax) in enumerate(batch_ranges):
        actual_batch_size = jmax - jmin
        profile_maxlen = max(len(tuples[j][0]) for j in range(jmin, jmax))
        profile_indexes_batch = np.full((actual_batch_size, profile_maxlen), 0, dtype=int)
        profile_size_batch = np.empty((actual_batch_size,))
        positive_index_batch = np.empty((actual_batch_size,), dtype=int)
        negative_index_batch = np.empty((actual_batch_size,), dtype=int)
        
        for j in range(actual_batch_size):
            # profile indexes
            for k,v in enumerate(tuples[jmin+j][0]):
                profile_indexes_batch[j][k] = v
            # profile size
            profile_size_batch[j] = len(tuples[jmin+j][0])        
            # positive index
            positive_index_batch[j] = tuples[jmin+j][1]
            # negative index
            negative_index_batch[j] = tuples[jmin+j][2]
            
        profile_indexes_batches[i] = profile_indexes_batch
        profile_size_batches[i] = profile_size_batch
        positive_index_batches[i] = positive_index_batch
        negative_index_batches[i] = negative_index_batch
        
    return dict(
        profile_indexes_batches = profile_indexes_batches,
        profile_size_batches    = profile_size_batches,
        positive_index_batches  = positive_index_batches,
        negative_index_batches  = negative_index_batches,
        n_batches               = n_batches,
    )


def sanity_check_minibatches(minibatches):
    profile_indexes_batches = minibatches['profile_indexes_batches']
    profile_size_batches = minibatches['profile_size_batches']
    positive_index_batches = minibatches['positive_index_batches']
    negative_index_batches = minibatches['negative_index_batches']
    n_batches = minibatches['n_batches']
    assert n_batches == len(profile_indexes_batches)
    assert n_batches == len(profile_size_batches)
    assert n_batches == len(positive_index_batches)
    assert n_batches == len(negative_index_batches)
    assert n_batches > 0
    
    for profile_indexes, profile_size, positive_index, negative_index in zip(
        profile_indexes_batches,
        profile_size_batches,
        positive_index_batches,
        negative_index_batches
    ):
        n = profile_size.shape[0]
        assert n == profile_indexes.shape[0]
        assert n == positive_index.shape[0]
        assert n == negative_index.shape[0]
        
        for i in range(n):
            assert positive_index[i] != negative_index[i]
            psz = int(profile_size[i])
            m = profile_indexes[i].shape[0]
            assert psz <= m
            for j in range(psz, m):
                assert profile_indexes[i][j] == 0


def train_network(train_minibatches, test_minibatches,
                  n_train_instances, n_test_instances, batch_size,
                  pretrained_embeddings,
                  user_layer_units,
                  item_layer_units,
                  profile_pooling_mode,
                  model_path,
                  max_seconds_training=3600,
                  min_seconds_to_check_improvement=60,
                  early_stopping_checks=4,
                  weight_decay=0.001,
                  learning_rates=[1e-3]):
    
    n_train_batches = train_minibatches['n_batches']
    n_test_batches = test_minibatches['n_batches']
    
    print('learning_rates = ', learning_rates)
    
    with tf.Graph().as_default():
        network = ContentBasedLearn2RankNetwork_Train(
            pretrained_embedding_dim=pretrained_embeddings.shape[1],
            user_layer_units=user_layer_units,
            item_layer_units=item_layer_units,
            weight_decay=weight_decay,
            profile_pooling_mode=profile_pooling_mode,
        )
        
        print('Variables to be trained:')
        for x in tf.global_variables():
            print('\t', x)            
        
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.99,
            allow_growth=True
        )
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            try:
                saver = tf.train.Saver()            
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                print('model successfully restored from checkpoint!')
            except ValueError:
                print('no checkpoint found: initializing variables with random values')
                os.makedirs(model_path, exist_ok=True)
                sess.run(tf.global_variables_initializer())            
            trainlogger = TrainLogger(model_path + 'train_logs.csv')

            # ========= BEFORE TRAINING ============
            
            initial_test_acc = 0.
            for profile_indexes, profile_size, positive_index, negative_index in zip(
                test_minibatches['profile_indexes_batches'],
                test_minibatches['profile_size_batches'],
                test_minibatches['positive_index_batches'],
                test_minibatches['negative_index_batches']
            ):
                minibatch_test_acc = network.get_test_accuracy(
                    sess, pretrained_embeddings, profile_indexes, profile_size, positive_index, negative_index)
                initial_test_acc += minibatch_test_acc
            initial_test_acc /= n_test_instances

            print("Before training: test_accuracy = %f" % initial_test_acc)
            
            best_test_acc = initial_test_acc
            seconds_training = 0
            elapsed_seconds_from_last_check = 0
            checks_with_no_improvement = 0
            last_improvement_loss = None
            
            # ========= TRAINING ============
            
            print ('Starting training ...')
            n_lr = len(learning_rates)
            lr_i = 0
            train_loss_ema = 0. # exponential moving average
            
            while seconds_training < max_seconds_training:
                
                for train_i, (profile_indexes, profile_size, positive_index, negative_index) in enumerate(zip(
                    train_minibatches['profile_indexes_batches'],
                    train_minibatches['profile_size_batches'],
                    train_minibatches['positive_index_batches'],
                    train_minibatches['negative_index_batches']
                )):
                    # optimize and get traing loss
                    start_t = time.time()
                    _, minibatch_train_loss = network.optimize_and_get_train_loss(
                        sess, learning_rates[lr_i], pretrained_embeddings, profile_indexes,
                        profile_size, positive_index, negative_index)
                    delta_t = time.time() - start_t
                    
                    # update train loss exponential moving average
                    train_loss_ema = 0.999 * train_loss_ema + 0.001 * minibatch_train_loss
                    
                    # update time tracking variables
                    seconds_training += delta_t
                    elapsed_seconds_from_last_check += delta_t
                    
                    # check for improvements using test set if it's time to do so
                    if elapsed_seconds_from_last_check >= min_seconds_to_check_improvement:
                        
                        # --- testing
                        test_acc = 0.
                        for _profile_indexes, _profile_size, _positive_index, _negative_index in zip(
                            test_minibatches['profile_indexes_batches'],
                            test_minibatches['profile_size_batches'],
                            test_minibatches['positive_index_batches'],
                            test_minibatches['negative_index_batches']
                        ):
                            minibatch_test_acc = network.get_test_accuracy(
                                sess, pretrained_embeddings, _profile_indexes,
                                _profile_size, _positive_index, _negative_index)                            
                            test_acc += minibatch_test_acc
                        test_acc /= n_test_instances
                    
                        print(("train_i=%d, train_loss = %.12f, test_accuracy = %.7f,"
                               " check_secs = %.2f, total_secs = %.2f") % (
                                train_i, train_loss_ema, test_acc, elapsed_seconds_from_last_check, seconds_training))                        
                        
                        # check for improvements
                        if (test_acc > best_test_acc) or (
                            test_acc == best_test_acc and (
                                last_improvement_loss is not None and\
                                last_improvement_loss > train_loss_ema
                            )
                        ):  
                            last_improvement_loss = train_loss_ema
                            best_test_acc = test_acc
                            checks_with_no_improvement = 0
                            saver = tf.train.Saver()
                            save_path = saver.save(sess, model_path)                    
                            print("   ** improvement detected: model saved to path ", save_path)
                            model_updated = True
                        else:
                            checks_with_no_improvement += 1                            
                            model_updated = False

                        # --- logging ---                        
                        trainlogger.log_update(
                            train_loss_ema, test_acc, n_train_instances, n_test_instances,
                            elapsed_seconds_from_last_check, batch_size, learning_rates[lr_i], 't' if model_updated else 'f')
                        
                        # --- check for early stopping
                        if checks_with_no_improvement >= early_stopping_checks:
                            if lr_i + 1 < len(learning_rates):
                                lr_i += 1
                                checks_with_no_improvement = 0
                                print("   *** %d checks with no improvements -> using a smaller learning_rate = %.8f" % (
                                    early_stopping_checks, learning_rates[lr_i]))
                            else:
                                print("   *** %d checks with no improvements -> early stopping :(" % early_stopping_checks)
                                return
                        
                        # --- reset check variables
                        elapsed_seconds_from_last_check = 0
            print('====== TIMEOUT ======')


train_minibatches = generate_minibatches(train_instances, max_users_items_per_batch=6000*10)
sanity_check_minibatches(train_minibatches)

test_minibatches = generate_minibatches(test_instances, max_users_items_per_batch=6000*10)
sanity_check_minibatches(test_minibatches)

learning_rates = get_decaying_learning_rates(1e-4, 1e-6, 0.6)

avg_train_batch_size = ceil(np.mean([b.shape[0] for b in train_minibatches['profile_indexes_batches']]))

# Training from scratch takes aprox. 5420 secs = 90 mins. Aprox. test acc should be 0.98, train loss 0.224
train_network(
    train_minibatches, test_minibatches,
    len(train_instances), len(test_instances),
    batch_size=avg_train_batch_size,
    pretrained_embeddings=concat_featmat,
    user_layer_units=[300,300,200],
    item_layer_units=[200,200],
    profile_pooling_mode='AVG+MAX',
    model_path = MODEL_PATH,
    max_seconds_training=3600 * 6,
    min_seconds_to_check_improvement=150,
    early_stopping_checks=2,
    weight_decay=.0001,
    learning_rates=learning_rates,
)
