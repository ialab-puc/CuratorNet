def load_embeddings_and_ids(embedding_path):
    import numpy as np
    featmat = np.load(embedding_path, allow_pickle=True)
    ids = featmat[:, 0]
    featmat = featmat[:, 1]
    newfeatmat = []
    for i in range(featmat.shape[0]):
        newfeatmat.append(featmat[i])
    id2index = {h:i for i, h in enumerate(ids)}
    index2id = {i:h for i, h in enumerate(ids)}
    
    return dict(featmat=np.array(newfeatmat),
                index2id=index2id,
                id2index=id2index)

def concatenate_featmats(artwork_ids, featmat_list, id2index_list):    
    assert len(featmat_list) == len(id2index_list)
    import numpy as np
    n = len(artwork_ids)
    m = sum(fm.shape[1] for fm in featmat_list)
    out_mat = np.empty(shape=(n,m))
    for i, _id in enumerate(artwork_ids):
        out_mat[i] = np.concatenate(
            [fm[id2index[_id]] for fm, id2index in zip(featmat_list, id2index_list)])
    return out_mat

class User:
    def __init__(self, uid):
        self._uid = uid
        self.artwork_ids = []
        self.artwork_idxs = []
        self.artwork_idxs_set = set()
        self.timestamps = []
        self.artist_ids_set = set()
        self.cluster_ids_set = set()
        
    def clear(self):
        self.artwork_ids.clear()
        self.artwork_idxs.clear()
        self.artwork_idxs_set.clear()        
        self.artist_ids_set.clear()
        self.cluster_ids_set.clear()
        self.timestamps.clear()
    
    def refresh_nonpurchased_cluster_ids(self, n_clusters):
        self.nonp_cluster_ids = [c for c in range(n_clusters) if c not in self.cluster_ids_set]
        assert len(self.nonp_cluster_ids) > 0
        
    def refresh_cluster_ids(self):
        self.cluster_ids = list(self.cluster_ids_set)
        assert len(self.cluster_ids) > 0
        
    def refresh_artist_ids(self):
        self.artist_ids = list(self.artist_ids_set)
        assert len(self.artist_ids) > 0
        
    def append_transaction(self, artwork_id, timestamp, artwork_id2index, artist_ids, cluster_ids):
        aidx = artwork_id2index[artwork_id]
        self.artwork_ids.append(artwork_id)
        self.artwork_idxs.append(aidx)
        self.artwork_idxs_set.add(aidx)
        self.artist_ids_set.add(artist_ids[aidx])
        self.cluster_ids_set.add(cluster_ids[aidx])
        self.timestamps.append(timestamp)
    
    def remove_last_nonfirst_purchase_basket(self, artwork_id2index, artist_ids, cluster_ids):
        baskets = self.baskets
        len_before = len(baskets)
        if len_before >= 2:
            last_b = baskets.pop()
            artwork_ids = self.artwork_ids[:last_b[0]]
            timestamps = self.timestamps[:last_b[0]]
            self.clear()
            for aid, t in zip(artwork_ids, timestamps):
                self.append_transaction(aid, t, artwork_id2index, artist_ids, cluster_ids)
            assert len(self.baskets) == len_before - 1
        
    def build_purchase_baskets(self):
        baskets = []
        prev_t = None
        offset = 0
        count = 0
        for i, t in enumerate(self.timestamps):
            if t != prev_t:
                if prev_t is not None:
                    baskets.append((offset, count))
                    offset = i
                count = 1
            else:
                count += 1
            prev_t = t
        baskets.append((offset, count))
        self.baskets = baskets
        
    def sanity_check_purchase_baskets(self):
        ids = self.artwork_ids
        ts = self.timestamps
        baskets = self.baskets        
        n = len(ts)
        assert(len(ids) == len(ts))
        assert(len(baskets) > 0)
        assert (n > 0)
        for b in baskets:
            for j in range(b[0], b[0] + b[1] - 1):
                assert(ts[j] == ts[j+1])
        for i in range(1, len(baskets)):
            b1 = baskets[i-1]
            b2 = baskets[i]
            assert(b1[0] + b1[1] == b2[0])
        assert(baskets[0][0] == 0)
        assert(baskets[-1][0] + baskets[-1][1] == n)
        
class VisualSimilarityHandler:
    def __init__(self, cluster_ids, embeddings):
        self._cluster_ids = cluster_ids
        self._cosineSimCache = dict()
        self.count = 0
        # store embeddings with l2 normalization
        from numpy.linalg import norm
        from numpy import reshape
        self._embeddings = embeddings / reshape(norm(embeddings, axis=1), (-1,1))
        
    def same(self,i,j):
        if self._cluster_ids[i] != self._cluster_ids[j]:
            return False        
        if abs(self.similarity(i,j) - 1.) < 1e-7:
            self.count += 1
            return True
        return False
    
    def similarity(self,i,j):
        if i > j:
            i, j = j, i
        k = (i,j)
        try:
            sim = self._cosineSimCache[k]
        except KeyError:
            from numpy import dot
            sim = self._cosineSimCache[k] = dot(self._embeddings[i], self._embeddings[j])
        return sim
    
    def validate_triple(self, q, p, n, margin=0.05):
        cq = self._cluster_ids[q]
        cp = self._cluster_ids[p]
        cn = self._cluster_ids[n]
        if cq == cp and cq != cn:
            return True
        if cq == cn and cq != cp:
            return False
        if self.similarity(q,p) > self.similarity(q,n) + margin:
            return True
        return False        
    
def get_decaying_learning_rates(maxlr, minlr, decay_coef):
    assert maxlr > minlr > 0
    assert 0 < decay_coef < 1
    lrs = []
    lr = maxlr
    while lr >= minlr:
        lrs.append(lr)
        lr *= decay_coef
    return lrs

def ground_truth_rank_indexes(ranked_inventory_ids, gt_ids_set):
    indexes = []
    for i, _id in enumerate(ranked_inventory_ids):
        if _id in gt_ids_set:
            indexes.append(i)
    return indexes

def auc_exact(ground_truth_indexes, inventory_size):
    n = len(ground_truth_indexes)
    assert inventory_size >= n
    if inventory_size == n:
        return 1
    auc = 0
    for i, idx in enumerate(ground_truth_indexes):
        auc += ((inventory_size - (idx+1)) - (n - (i+1))) / (inventory_size - n)
    try:
        auc /= n
    except ZeroDivisionError:
        auc = 0
    return auc
