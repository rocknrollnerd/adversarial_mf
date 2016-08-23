# import pyximport
# pyximport.install()

import lasagne
import numpy as np
import sys
from mf import MF
from utils import get_data
from adversarial_mf import AdversarialMF
from wrmf import WRMF
from popular import Popular
from sklearn.grid_search import GridSearchCV
import metrics
from sklearn.cross_validation import train_test_split, ShuffleSplit

import matplotlib.pyplot as plt
from collections import Counter
from scipy import sparse

import time

import logging
logging.basicConfig(level=logging.DEBUG)


def sparse_to_x(mat):
    mat = sparse.coo_matrix(mat)
    x = np.float32(np.vstack([
        mat.row, mat.col, mat.data
    ])).T
    # np.random.shuffle(x)
    return x


def zero_center(data):
    # consider nonzero values only
    idx = data != 0
    mmax = data[idx].max()
    mmin = data[idx].min()
    data[idx] -= (mmax + mmin) / 2.
    return data


if __name__ == '__main__':
    # data = zero_center(get_data(sys.argv[1]))
    data = get_data(sys.argv[1])
    X = sparse_to_x(data)
    X_train, X_test = train_test_split(X, test_size=0.1)
    print len(X)

    rec_cls = {
        'mf': MF,
        'adversarial_mf': AdversarialMF,
        'wrmf': WRMF,
        'popular': Popular
    }.get(sys.argv[2])

    kwargs = {}
    rec = rec_cls(n_users=data.shape[0], n_items=data.shape[1], **kwargs)

    # mf param grid
    # parameters = {
        # 'n_factors': [20],
        # 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
        # 'optimizer': [lasagne.updates.adam, lasagne.updates.adagrad],
        # 'regularization': [0.00015, 0.001, 0.01, 0.00001, 0.0002, 0.05, 0.1],
    # }

    # wrmf param grid
    parameters = {
        # 'n_factors': [20],
        # 'regularization': [0.1],
        # 'alpha': [1]
    }

    # amf param grid
    # parameters = {
        # 'n_factors': [20],
        # 'g_lr': [0.01, 0.1, 0.001],
        # 'd_lr': [0.01, 0.1, 0.001],
        # 'discriminator_k': [1, 10],
        # 'discriminator_regularization': [0.01, 0.1, 0.2, 0.5]
    # }

    t = time.time()
    search = GridSearchCV(rec, parameters, n_jobs=1, scoring=metrics.avg_roc_auc, cv=ShuffleSplit(len(X), 3))
    search.fit(X)

    print 'fitted in', time.time() - t
    print search.grid_scores_
    print search.best_params_, search.best_score_
