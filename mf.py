import theano
import theano.tensor as T
import lasagne
import numpy as np
from sklearn.cross_validation import train_test_split

from base import BaseRecommender


class MF(BaseRecommender):

    def __init__(self, n_users, n_items, n_factors=20, learning_rate=0.2, optimizer=lasagne.updates.adagrad, regularization=0.0002, n_epochs=100, verbose=False):
        super(MF, self).__init__(n_users, n_items)
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.verbose = verbose

    def initialize(self):
        users = T.ivector()
        items = T.ivector()
        ratings = T.vector()

        self.U = theano.shared(
            np.array(
                np.random.normal(scale=0.001, size=(self.n_users, self.n_factors)),
                dtype=theano.config.floatX
            )
        )
        self.I = theano.shared(
            np.array(
                np.random.normal(scale=0.001, size=(self.n_items, self.n_factors)),
                dtype=theano.config.floatX
            )
        )

        predictions = (self.U[users] * self.I[items]).sum(axis=1)

        train_error = (
            ((predictions - ratings) ** 2).mean() +
            self.regularization * (
                T.sum(self.U ** 2) +
                T.sum(self.I ** 2)
            )
        )
        test_error = ((predictions - ratings) ** 2).mean()

        params = [self.U, self.I]
        learning_rate = theano.shared(np.array(self.learning_rate, dtype=theano.config.floatX))
        updates = self.optimizer(train_error, params, learning_rate=learning_rate)
        self.train_theano = theano.function([users, items, ratings], train_error, updates=updates)
        self.test_theano = theano.function([users, items, ratings], test_error)
        self.predict_theano = theano.function([users, items], predictions)

    def fit(self, X, y=None):
        self.initialize()
        M = self.construct_sparse_matrix(X).tocoo()

        users = np.int32(M.row)
        items = np.int32(M.col)
        ratings = np.float32(M.data)

        idx = np.arange(len(users))
        train_idx, test_idx = train_test_split(idx, test_size=0.1)

        train_users = users[train_idx]
        train_items = items[train_idx]
        train_ratings = ratings[train_idx]

        validation_users = users[test_idx]
        validation_items = items[test_idx]
        validation_ratings = ratings[test_idx]

        for e in xrange(self.n_epochs):
            train_err = np.sqrt(float(self.train_theano(train_users, train_items, train_ratings)))
            validation_err = np.sqrt(float(self.test_theano(validation_users, validation_items, validation_ratings)))
            if self.verbose:
                print 'epoch {}, train RMSE: {:.6f}, validation RMSE {:.6f}'.format(e, train_err, validation_err)
        self.popularity_ = self.count_popularity(train_items)
        return self

    def predict(self, users, items):
        return self.predict_theano(np.int32(users), np.int32(items))
