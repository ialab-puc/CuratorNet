import math
from itertools import combinations
import numpy as np

def precision(ground_truth_indexes, k):
    count = 0
    for i in ground_truth_indexes:
        if i >= k: break
        count += 1
    return count / k

def recall(ground_truth_indexes, k):
    count = 0
    for i in ground_truth_indexes:
        if i >= k: break
        count += 1
    return count / len(ground_truth_indexes)

def f1score(ground_truth_indexes, k):
    count = 0
    for i in ground_truth_indexes:
        if i >= k: break
        count += 1
    p = count / k
    r = count / len(ground_truth_indexes)
    return 2 * p * r / (p + r) if (p + r) > 0.0 else 0.0

# pre-compute ideal DCGs for performance improvement
IDEAL_DCG = np.zeros((1000,))
IDEAL_DCG[0] = 0
for _i in range(1, 1000):
    IDEAL_DCG[_i] = IDEAL_DCG[_i-1] + 1/math.log2(_i+1)
    
def nDCG(ground_truth_indexes, k):
    dcg = 0
    count = 0
    for i in ground_truth_indexes:
        if i >= k: break
        dcg += 1 / math.log2(i+2)
        count += 1
    return dcg / IDEAL_DCG[count] if dcg > 0 else 0

def average_precision(ground_truth_indexes, k):
    tot = 0
    hits = 0
    for i in ground_truth_indexes:
        if i >= k: break
        hits += 1
        tot += hits / (i+1)
    return tot / k

def auc_exact(ground_truth_indexes, inventory_size):
    n = len(ground_truth_indexes)
    assert inventory_size >= n
    if inventory_size == n:
        return 1
    auc = 0
    for i, idx in enumerate(ground_truth_indexes):
        auc += ((inventory_size - (idx+1)) - (n - (i+1))) / (inventory_size - n)
    auc /= n
    return auc

def reciprocal_rank(first_relevant_index):
    return 1 / (first_relevant_index + 1)

def jaccard_index(set1, set2):
    x = sum(1 for e in set1 if e in set2)
    return x / (len(set1) + len(set2) - x)