import tensorflow as tf


class TrainLogger:
    def __init__(self, filepath):
        self._filepath = filepath
        try:
            with open(filepath) as f: pass                
        except FileNotFoundError:
            with open(filepath, 'w') as f:
                f.write(','.join(['train_loss', 'validation_accuracy',
                          'n_train_tuples', 'n_validation_tuples', 'elapsed_seconds',
                          'batch_size', 'learning_rate', 'model_updated']) + '\n')
    def log_update(self, train_loss, validation_accuracy, n_train_tuples, n_validation_tuples,
                 elapsed_seconds, batch_size, learning_rate, model_updated):
            with open(self._filepath, 'a') as f:
                f.write('%.10f,%.5f,%d,%d,%.5f,%d,%f,%s\n' % (train_loss, validation_accuracy,
                        n_train_tuples, n_validation_tuples, elapsed_seconds, batch_size, learning_rate,
                        model_updated))


class CuratorNet_Base:
    
    @staticmethod
    def compute_user_embedding(profile_aggregation_vector, user_layer_units):
        last_output = profile_aggregation_vector
        n_layers = len(user_layer_units)
        
        # hidden layers
        for i in range(n_layers-1):
            last_output = tf.layers.dense(
                inputs=last_output,
                units=user_layer_units[i],
                activation=tf.nn.selu,
                name='user_hidden_%d' % (i+1)
            )
        
        # user final vector
        return tf.layers.dense(
            inputs=last_output,
            units=user_layer_units[-1],
            activation=tf.nn.selu,
            name='user_vector'
        )
    
    @staticmethod
    def compute_item_embedding(X, item_layer_units):
        with tf.variable_scope("trainable_item_embedding", reuse=tf.AUTO_REUSE):
            last_output = X
            for i, units in enumerate(item_layer_units):
                last_output = tf.layers.dense(
                    inputs=last_output,
                    units=units,
                    activation=tf.nn.selu,
                    name='fc%d' % (i+1)
                )
            return last_output


class CuratorNet_Train(CuratorNet_Base):
    def __init__(self, pretrained_embedding_dim, user_layer_units, item_layer_units, weight_decay,
                 profile_pooling_mode='AVG'):
        
        assert user_layer_units[-1] == item_layer_units[-1]
        
        # --- placeholders
        self._pretrained_embeddings = tf.placeholder(shape=[None, pretrained_embedding_dim],
                                                     dtype=tf.float32,
                                                     name='pretrained_embeddings')            
        self._profile_item_indexes = tf.placeholder(shape=[None,None], dtype=tf.int32,
                                                    name='profile_item_indexes')
        self._profile_sizes = tf.placeholder(shape=[None], dtype=tf.float32,
                                                   name='profile_sizes')        
        self._positive_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='positive_item_index')
        self._negative_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='negative_item_index')
        self._learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

        # ---- aggregate user profile and output user embedding
        
        # profile item embeddings
        tmp = tf.gather(self._pretrained_embeddings, self._profile_item_indexes)
        self._profile_item_embeddings = self.compute_item_embedding(tmp, item_layer_units)
        
        # avgpool masking
        self._profile_masks__avgpool = tf.expand_dims(tf.sequence_mask(self._profile_sizes, dtype=tf.float32), -1)        
        self._masked_profile_item_embeddings__avgpool =\
            self._profile_item_embeddings * self._profile_masks__avgpool
        
        # maxpool masking
        self._profile_masks__maxpool = (1. - self._profile_masks__avgpool) * -9999.
        self._masked_profile_item_embeddings__maxpool =\
            self._masked_profile_item_embeddings__avgpool + self._profile_masks__maxpool
        
        # items avgpool
        self._profile_items_avgpool =\
            tf.reduce_sum(self._masked_profile_item_embeddings__avgpool, axis=1) /\
            tf.reshape(self._profile_sizes, [-1, 1])
        
        # items maxpool
        self._profile_items_maxpool =\
            tf.reduce_max(self._masked_profile_item_embeddings__maxpool, axis=1)
        
        # user vector
        if profile_pooling_mode == 'AVG':
            profile_aggregation_vector = self._profile_items_avgpool            
        else:
            assert profile_pooling_mode == 'AVG+MAX'
            profile_aggregation_vector = tf.concat([
                self._profile_items_avgpool,
                self._profile_items_maxpool], 1)
        self._user_vector = self.compute_user_embedding(
            profile_aggregation_vector,
            user_layer_units,
        )
        
        # ---- positive item vector
        tmp = tf.gather(self._pretrained_embeddings, self._positive_item_index)
        self._positive_item_vector = self.compute_item_embedding(tmp, item_layer_units)
        
        # ---- negative item vector
        tmp = tf.gather(self._pretrained_embeddings, self._negative_item_index)
        self._negative_item_vector = self.compute_item_embedding(tmp, item_layer_units)
        
        # --- train loss
        
        # ranking loss
        dot_pos = tf.reduce_sum(tf.multiply(self._user_vector, self._positive_item_vector), 1)
        dot_neg = tf.reduce_sum(tf.multiply(self._user_vector, self._negative_item_vector), 1)
        dot_delta = dot_pos - dot_neg
        ones = tf.fill(tf.shape(self._user_vector)[:1], 1.0)
        rank_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dot_delta, labels=ones)
        rank_loss = tf.reduce_mean(rank_loss, name='train_loss')
        
        if weight_decay > 0:
            # l2 loss
            _vars = [v for v in tf.trainable_variables() if 'bias' not in v.name]        
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in _vars ])        
            # train loss
            self._train_loss = rank_loss + weight_decay * l2_loss
        else:
            assert weight_decay == 0
            # train loss
            self._train_loss = rank_loss
        
        # --- test accuracy
        accuracy = tf.reduce_sum(tf.cast(dot_delta > .0, tf.float32), name = 'test_accuracy')
        self._test_accuracy = accuracy
        
        # --- optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._train_loss)
    
    def optimize_and_get_train_loss(self, sess, learning_rate, pretrained_embeddings,
                                    profile_item_indexes, profile_sizes,
                                    positive_item_index, negative_item_index):
        return sess.run([
            self._optimizer,
            self._train_loss,
        ], feed_dict={
            self._learning_rate: learning_rate,
            self._pretrained_embeddings: pretrained_embeddings,
            self._profile_item_indexes: profile_item_indexes,
            self._profile_sizes: profile_sizes,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
    def get_test_accuracy(self, sess, pretrained_embeddings, profile_item_indexes, profile_sizes,
             positive_item_index, negative_item_index):
        return sess.run(
            self._test_accuracy, feed_dict={
            self._pretrained_embeddings: pretrained_embeddings,
            self._profile_item_indexes: profile_item_indexes,
            self._profile_sizes: profile_sizes,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })


class CuratorNet_Precomputation(CuratorNet_Base):
    def __init__(self, pretrained_embedding_dim, item_layer_units):        
        
        # --- placeholders
        self._pretrained_embeddings = tf.placeholder(shape=[None, pretrained_embedding_dim], dtype=tf.float32)
        
        # --- item vectors
        self._item_vectors = self.compute_item_embedding(self._pretrained_embeddings, item_layer_units)
        
    def precompute_tensors(self, sess, pretrained_embeddings):
        return sess.run(self._item_vectors, feed_dict={
            self._pretrained_embeddings: pretrained_embeddings,
        })


class CuratorNet_Evaluation(CuratorNet_Base):
    def __init__(self, user_layer_units, latent_space_dim, profile_pooling_mode='AVG'):
        
        assert user_layer_units[-1] == latent_space_dim
        
        # --- placeholders
        self._precomputed_item_vectors = tf.placeholder(shape=[None, latent_space_dim], dtype=tf.float32)
        self._profile_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
        self._candidate_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
            
        # ---- user profile vector
        
        tmp = tf.gather(self._precomputed_item_vectors, self._profile_item_indexes) 
        
        # profile items avgpool
        self._profile_items_avgpool = tf.reshape(tf.reduce_mean(tmp, axis=0), (1, latent_space_dim))
        
        # profile items maxpool
        self._profile_items_maxpool = tf.reshape(tf.reduce_max(tmp, axis=0), (1, latent_space_dim))
        
        # user vector
        if profile_pooling_mode == 'AVG':
            profile_aggregation_vector = self._profile_items_avgpool            
        else:
            assert profile_pooling_mode == 'AVG+MAX'
            profile_aggregation_vector = tf.concat([
                self._profile_items_avgpool,
                self._profile_items_maxpool], 1)
        self._user_vector = self.compute_user_embedding(
            profile_aggregation_vector,
            user_layer_units,
        )
        
        # ---- candidate item vectors
        self._candidate_item_vectors = tf.gather(self._precomputed_item_vectors,
                                                 self._candidate_item_indexes)
        
        # ---- match scores
        self._match_scores = tf.reduce_sum(self._user_vector * self._candidate_item_vectors, 1)
    
    def get_match_scores(self, sess, precomputed_item_vectors, profile_item_indexes, candidate_items_indexes):
        return sess.run(
            self._match_scores, feed_dict={
            self._precomputed_item_vectors: precomputed_item_vectors,
            self._profile_item_indexes: profile_item_indexes,
            self._candidate_item_indexes: candidate_items_indexes,
        })


class VBPR_Network_Base:
    
    @staticmethod
    def trainable_image_embedding(X, output_dim):
        with tf.variable_scope("trainable_image_embedding", reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense( # None -> output_dim
                inputs=X,
                units=output_dim,
                name='fc1',
                use_bias=False,
                activation=None,
            )
            return fc1


class VBPR_Network_Train(VBPR_Network_Base):
    def __init__(self, n_users, n_items, user_latent_dim, item_latent_dim, item_visual_dim, pretrained_dim, weight_decay):
        
        assert (user_latent_dim == item_latent_dim + item_visual_dim)
        
        self._item_visual_dim = item_visual_dim
        
        # --- placeholders
        self._pretrained_image_embeddings = tf.placeholder(shape=[None, pretrained_dim], dtype=tf.float32,
                                                     name='pretrained_image_embeddings')    
        self._user_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name='user_index')
        self._positive_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='positive_item_index')
        self._negative_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='negative_item_index')
        self._learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
            
        # ------------------------------------
        # ---- Global trainable variables
        
        # -- user latent factor matrix
        # (n_users x user_latent_dim)
        self._user_latent_factors = tf.Variable(
            tf.random_uniform([n_users, user_latent_dim], -1.0, 1.0),
            name='user_latent_factors'
        )
        
        # -- item latent factor matrix
        # (n_items x item_latent_dim)
        self._item_latent_factors = tf.Variable(
            tf.random_uniform([n_items, item_latent_dim], -1.0, 1.0),
            name='item_latent_factors'
        )
        
        # -- item latent biases
        self._item_latent_biases = tf.Variable(
            tf.random_uniform([n_items], -1.0, 1.0),
            name='item_latent_biases'
        )
        
        # -- global visual bias
        self._visual_bias = tf.Variable(
            tf.random_uniform([pretrained_dim], -1.0, 1.0),
            name='visual_bias'
        )
        
        # -------------------------------
        # ---- minibatch tensors
        
        # -- user
        self._user_latent_vector = tf.gather(self._user_latent_factors, self._user_index)
        
        # -- positive item
        self._pos_vector,\
        self._pos_latent_bias,\
        self._pos_visual_bias = self.get_item_variables(self._positive_item_index)
        self._pos_score = tf.reduce_sum(self._user_latent_vector * self._pos_vector, 1) +\
                    self._pos_latent_bias +\
                    self._pos_visual_bias
        
        # -- negative item
        self._neg_vector,\
        self._neg_latent_bias,\
        self._neg_visual_bias = self.get_item_variables(self._negative_item_index)
        self._neg_score = tf.reduce_sum(self._user_latent_vector * self._neg_vector, 1) +\
                    self._neg_latent_bias +\
                    self._neg_visual_bias
        
        # -------------------------------
        # ---- train-test tensors
        
        # -- train loss
        
        # ranking loss
        delta_score = self._pos_score - self._neg_score
        ones = tf.fill(tf.shape(self._user_latent_vector)[:1], 1.0)
        rank_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=delta_score, labels=ones)
        rank_loss = tf.reduce_mean(rank_loss, name='rank_loss')
        
        if weight_decay > 0:
            # l2 loss
            _vars = [v for v in tf.trainable_variables() if 'bias' not in v.name]        
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in _vars ])        
            # train loss
            self._train_loss = rank_loss + weight_decay * l2_loss
        else:
            assert weight_decay == 0
            # train loss
            self._train_loss = rank_loss
        
        # -- test accuracy
        accuracy = tf.reduce_sum(tf.cast(delta_score > .0, tf.float32), name='test_accuracy')
        self._test_accuracy = accuracy
        
        # -- optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._train_loss)
        
    def get_item_variables(self, item_index):
        pre_vector = tf.gather(self._pretrained_image_embeddings, item_index)
        # 1) item vector
        #    1.1) visual vector
        visual_vector = self.trainable_image_embedding(pre_vector, self._item_visual_dim)
        #    1.2) latent vector
        latent_vector = tf.gather(self._item_latent_factors, item_index)
        #    1.3) concatenation
        final_vector = tf.concat([visual_vector, latent_vector], 1)
        # 2) latent bias
        latent_bias = tf.gather(self._item_latent_biases, item_index)
        # 3) visual bias
        visual_bias = tf.reduce_sum(pre_vector * self._visual_bias, 1)
        # return
        return final_vector, latent_bias, visual_bias
    
    def optimize_and_get_train_loss(self, sess, pretrained_image_embeddings, user_index, positive_item_index,
                                    negative_item_index, learning_rate):
        return sess.run([
            self._optimizer,
            self._train_loss,
        ], feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
            self._learning_rate: learning_rate,
        })
    
    def get_train_loss(self, sess, pretrained_image_embeddings, user_index, positive_item_index, negative_item_index):
        return sess.run( 
            self._train_loss, feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
    def get_test_accuracy(self, sess, pretrained_image_embeddings, user_index, positive_item_index, negative_item_index):
        return sess.run(
            self._test_accuracy, feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
class VBPR_Network_Evaluation(VBPR_Network_Base):
    def __init__(self, n_users, n_items, user_latent_dim, item_latent_dim, item_visual_dim,
                 pretrained_dim=2048):
        
        # --- placeholders
        self._pretrained_image_embeddings = tf.placeholder(shape=[None, pretrained_dim], dtype=tf.float32)
        self._item_index = tf.placeholder(shape=[None], dtype=tf.int32)
            
        # ------------------------------------
        # ---- Global trainable variables
        
        # -- user latent factor matrix
        # (n_users x user_latent_dim)
        self._user_latent_factors = tf.Variable(
            tf.random_uniform([n_users, user_latent_dim], -1.0, 1.0),
            name='user_latent_factors'
        )
        
        # -- item latent factor matrix
        # (n_items x item_latent_dim)
        self._item_latent_factors = tf.Variable(
            tf.random_uniform([n_items, item_latent_dim], -1.0, 1.0),
            name='item_latent_factors'
        )
        
        # -- item latent biases
        self._item_latent_biases = tf.Variable(
            tf.random_uniform([n_items], -1.0, 1.0),
            name='item_latent_biases'
        )
        
        # -- global visual bias
        self._visual_bias = tf.Variable(
            tf.random_uniform([pretrained_dim], -1.0, 1.0),
            name='visual_bias'
        )
        
        # -------------------------------
        # ---- minibatch tensors
        
        item_pre_vector = tf.gather(self._pretrained_image_embeddings, self._item_index)
        
        # 1) item vector
        #    1.1) visual vector
        item_visual_vector = self.trainable_image_embedding(item_pre_vector, item_visual_dim)
        #    1.2) latent vector
        item_latent_vector = tf.gather(self._item_latent_factors, self._item_index)
        #    1.3) concatenation
        self._item_final_vector = tf.concat([item_visual_vector, item_latent_vector], 1)
        
        # 2) item bias
        #    1.1) visual bias
        item_visual_bias = tf.reduce_sum(item_pre_vector * self._visual_bias, 1)
        #    1.2) latent bias
        item_latent_bias = tf.gather(self._item_latent_biases, self._item_index)
        #    1.3) final bias
        self._item_final_bias = item_visual_bias + item_latent_bias
    
    def get_item_final_vector_bias(self, sess, pretrained_image_embeddings, item_index):
        return sess.run([
            self._item_final_vector,
            self._item_final_bias,
        ], feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._item_index: item_index,
        })
    
    def get_user_latent_vectors(self, sess):
        return sess.run(self._user_latent_factors)