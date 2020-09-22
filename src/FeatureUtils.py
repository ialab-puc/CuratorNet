import json
import os.path as ospath
import numpy as np

def read_ids_file(dirpath, ids_filename):
    filepath = ospath.join(dirpath, ids_filename)
    with open(filepath) as f:
        index2id = [x.strip() for x in f.readlines()]
    id2index = {_id:i for i, _id in enumerate(index2id)}
    return index2id, id2index

def read_numpy_features_matrix(dirpath, feat_filename, ids_filename):
    features = np.load(ospath.join(dirpath, feat_filename), allow_pickle=True)
    index2id, id2index = read_ids_file(dirpath, ids_filename)
    return dict(
        features=features,
        index2id=index2id,
        id2index=id2index,
    )

def numpy_array_feature_matrix(feature_matrix, feature_id2index, artworks):
    first = True
    for artwork in artworks:
        i = feature_id2index[artwork.id]
        feat_vec = feature_matrix[i]
        if first:
            output_matrix = np.empty((len(artworks), len(feat_vec)))
            first = False
        output_matrix[artwork.index] = feat_vec
    return output_matrix