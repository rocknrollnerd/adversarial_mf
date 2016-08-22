from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
import numpy as np

from utils import bincount_relative


class BaseRecommender(BaseEstimator):

    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items

    def construct_sparse_matrix(self, X):
        """
        X is supposed to be shaped as n x 3
        array, where:
            * X[:, 0] contains user indices
            * X[:, 1] contains item indices
            * X[:, 2] contains ratings (values)
        Combined with self.n_users and self.n_items
        X can be converted to sparse matrix which
        is the preferred way to go with recommender problems.
        """
        assert len(X.shape) == 2 and X.shape[1] == 3, 'Expecting `n x 3` sized array'
        return csr_matrix(
            (X[:, 2], (np.int32(X[:, 0]), np.int32(X[:, 1]))), shape=(self.n_users, self.n_items)
        )

    def count_popularity(self, train_items):
        return bincount_relative(train_items, min_value=0, max_value=self.n_items - 1)
