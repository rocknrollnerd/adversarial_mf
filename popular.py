from base import BaseRecommender
from utils import bincount_relative

from operator import itemgetter
import numpy as np


class Popular(BaseRecommender):
    """
    Simple `most popular` recommender.
    """

    def fit(self, X, y=None):
        items = np.int32(X[:, 1])

        popularity = dict(zip(
            np.arange(0, self.n_items),
            bincount_relative(items, max_value=self.n_items - 1, min_value=0)
        ))
        # for top-n prediction, also store two sorted lists rather than a dict
        items, scores = zip(*sorted(popularity.items(), reverse=True, key=itemgetter(1)))
        self.popularity_ = popularity
        self.items_ = items
        self.scores_ = scores
        return self

    def predict(self, users, items):
        # no personalization; just return popularity for requested items
        return np.array([self.popularity_[i] for i in items])
