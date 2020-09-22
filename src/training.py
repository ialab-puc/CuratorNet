import os
import time
import numpy as np
import tensorflow as tf
from Networks import *

"""
Training loop utilities 
"""

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
                  epochs=100,
                  early_stopping_checks=4,
                  weight_decay=0.001,
                  learning_rates=[1e-3]):
    
    n_train_batches = train_minibatches['n_batches']
    n_test_batches = test_minibatches['n_batches']
    
    print('learning_rates = ', learning_rates)
    
    with tf.Graph().as_default():
        network = CuratorNet_Train(
            pretrained_embedding_dim=pretrained_embeddings.shape[1],
            user_layer_units=user_layer_units,
            item_layer_units=item_layer_units,
            weight_decay=weight_decay,
            profile_pooling_mode=profile_pooling_mode,
        )
        
        print('Variables to be trained:')
        for x in tf.global_variables():
            print('\t', x)
            
        with tf.Session() as sess:
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
            batches_passed = 0
            epochs_passed = 0
            
            # ========= TRAINING ============
            
            print ('Starting training ...')
            n_lr = len(learning_rates)
            lr_i = 0
            train_loss_ema = 0. # exponential moving average
            
            while epochs_passed < epochs:
                
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
                    batches_passed += 1
                    
                    # check for improvements using test set if it's time to do so
                    if batches_passed == train_minibatches['n_batches']:  
                        epochs_passed += 1
                        batches_passed = 0  
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
                    
                        print(("EPOCH=%d => train_i=%d, train_loss = %.12f, test_accuracy = %.7f,"
                               " epoch_secs = %.2f, total_secs = %.2f") % (
                                epochs_passed, train_i, train_loss_ema, test_acc, elapsed_seconds_from_last_check, seconds_training))                        
                        
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
