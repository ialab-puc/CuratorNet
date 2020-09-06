def get_id2url_map():
    import urllib
    
    id2url = dict()
    with open('./artwork_ids.txt') as f:
        for line in f.readlines():
            line = line.rstrip()
            _id, url = line.split(' ', 1)
            assert(url.index(_id) > 0)
            _id = int(_id)
            if url.index('static') == 0:
                url = 'http://' + url
            idx = url.index('Images/') + len('Images/')
            url = url[:idx] + urllib.parse.quote(url[idx:])
            id2url[_id] = url
    return id2url

def silentremove(filename):
    import os
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def download_image(url, outpath):
    import requests
    import shutil
    try:
        r = requests.get(url, stream=True, timeout=3)
        if r.status_code == 200:
            with open(outpath, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        return True
    except requests.exceptions.Timeout as e:
        print('Timeout detected: url=',url)
        print(e)
        silentremove(outpath)
    except Exception as e:
        print('Unexpected exception detected: url=',url)
        print(e)
        silentremove(outpath)
    finally:
        r.close()
    return False

def process_image_batch(
        models, featmats, image_files, image_ids, id2url_map,
        i_start, i_end, preprocess_input_fn, image_target_size=(224, 224)):
    assert len(models) > 0 and len(models) == len(featmats)
    assert 0 <= i_start < i_end <= len(image_files)
    
    import numpy as np
    from keras.preprocessing import image
    
    n = i_end - i_start
    batch_X = np.empty(shape=(n, *image_target_size, 3))
    for i in range(i_start, i_end):
        file = image_files[i]
        img_loaded = False
        tries = 0
        while 1:
            try:
                img = image.load_img(file, target_size=image_target_size)
                img_loaded = True
                break
            except OSError as e:
                print('OSError detected, file = ', file)                
                tries += 1                
                if (tries == 3):
                    break
                url = id2url_map[image_ids[i]]
                print('(attempt %d) we will try to download the image from url=%s' % (tries, url))
                if download_image(url, file):
                    print('image successfully downloaded to %s!' % file)
        if not img_loaded:            
            raise Exception('failed to load image file=%s, url=%s' % (file, url))
        batch_X[i - i_start] = image.img_to_array(img)
    batch_X = preprocess_input_fn(batch_X)
    for model, featmat in zip(models, featmats):
        featmat[i_start:i_end] = model.predict(batch_X)
        
def get_image(image_cache, _id):
    try:
        return image_cache[_id]
    except KeyError:
        from PIL import Image
        img = Image.open('/mnt/workspace/Ugallery/images/%d.jpg' % _id)
        image_cache[_id] = img
        return img

def plot_images(plt, image_cache, ids):    
    plt.close()
    n = len(ids)    
    nrows = n//5 + int(n%5>0)
    ncols = min(n, 5)
    plt.figure(1, (20, 5 * nrows))
    for i, _id in enumerate(ids):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        img = get_image(image_cache, _id)
        ax.set_title('%d) id = %d' % (i, _id))
        ax.imshow(img, interpolation="nearest")
    plt.show()
    
    
def read_ids_file(dirpath, ids_filename):
    from os import path    
    filepath = path.join(dirpath, ids_filename)
    if ids_filename[-5:] == '.json':
        with open(filepath) as f:
            index2id = json.load(f)
    elif ids_filename[-4:] == '.npy':
        import numpy as np
        index2id = np.load(filepath, allow_pickle=True)
    else:
        assert ids_filename[-3:] == 'ids'
        with open(filepath) as f:
            index2id = [int(x) for x in f.readlines()]
    id2index = {_id:i for i, _id in enumerate(index2id)}
    return index2id, id2index

def load_embeddings_and_ids(dirpath, embedding_file, ids_file):
    import numpy as np
    from os import path
    featmat = np.load(path.join(dirpath, embedding_file), allow_pickle=True)
    index2id, id2index = read_ids_file(dirpath, ids_file)
    return dict(
        featmat=featmat,
        index2id=index2id,
        id2index=id2index,
    )

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
