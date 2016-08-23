import implicit
import numpy as np
from scipy.sparse import csr_matrix

from base import BaseRecommender


class WRMF(BaseRecommender):

    def __init__(self, n_users, n_items, n_factors=20, regularization=0.0002, n_epochs=50, verbose=False, alpha=40):
        super(WRMF, self).__init__(n_users, n_items)
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.alpha = alpha

    def fit(self, X, y=None):
        M = self.construct_sparse_matrix(X).tocoo()
        items = np.int32(M.col)
        self.popularity_ = self.count_popularity(items)
        M = M.toarray()
        M = M * self.alpha
        M = csr_matrix(M).astype('double')
        self.U, self.I = implicit.alternating_least_squares(M, factors=self.n_factors, regularization=self.regularization, iterations=self.n_epochs)
        return self

    def predict(self, users, items):
        return (self.U[np.int32(users)] * self.I[np.int32(items)]).sum(axis=1)
