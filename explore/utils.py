import numpy as np
from scipy.stats import rankdata

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='min', axis=1)
    filter_rank = 1
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks[ranks == 0] = np.inf
    ranks = np.min(ranks, axis=1)
    return list(ranks)

def cal_top1(scores, labels, filters):
    top1_indices = np.argmax(scores, axis=1)
    predicted = labels[np.arange(labels.shape[0]), top1_indices]
    predicted[predicted==0] = 20
    return list(predicted)

def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_3 = sum(ranks<=3) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1,h_3, h_10
