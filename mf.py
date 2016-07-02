import theano
import theano.tensor as T
import lasagne
import numpy as np
from scipy import sparse

from utils import get_data
from metrics import mean_average_precision, roc_auc


class ClassicMF(object):

    def __init__(self, num_factors=20, learning_rate=0.01, optimizer=lasagne.updates.adagrad, regularization=0.00015, num_epochs=100):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.regularization = regularization
        self.num_epochs = num_epochs

    def initialize(self, train, test):
        num_users, num_items = train.shape
        self.num_users, self.num_items = num_users, num_items
        users = T.ivector()
        items = T.ivector()
        ratings = T.vector()

        self.U = theano.shared(
            np.array(
                np.random.normal(scale=0.01, size=(num_users, self.num_factors)),
                dtype=theano.config.floatX
            )
        )
        self.I = theano.shared(
            np.array(
                np.random.normal(scale=0.01, size=(num_items, self.num_factors)),
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
        self.train = theano.function([users, items, ratings], train_error, updates=updates)
        self.test = theano.function([users, items, ratings], test_error)
        self.predict = theano.function([users, items], predictions)

    def train(self, train, test):
        self.initialize(train, test)

        train_coo = sparse.coo_matrix(train)
        test_coo = sparse.coo_matrix(test)

        train_users = np.int32(train_coo.row)
        train_items = np.int32(train_coo.col)
        train_ratings = np.float32(train_coo.data)
        test_users = np.int32(test_coo.row)
        test_items = np.int32(test_coo.col)
        test_ratings = np.float32(test_coo.data)

        train_errors = []
        test_errors = []
        prev_err = None

        for e in xrange(self.num_epochs):
            train_err = self.train(train_users, train_items, train_ratings)
            test_err = self.test(test_users, test_items, test_ratings)
            if prev_err is not None:
                if test_err > prev_err:
                    print 'early stopping'
                    break
            prev_err = test_err
            # rmse
            train_err = np.sqrt(float(train_err))
            test_err = np.sqrt(float(test_err))
            train_errors.append(train_err)
            test_errors.append(test_err)

            test_predictions = self.predict(test_users, test_items)
            roc_auc_val = roc_auc(test_ratings, test_predictions)
            map_val = mean_average_precision(test_ratings, test_predictions)
            print 'epoch {}, train RMSE: {:.6f}, test RMSE: {:.6f}, ROC AUC: {:.4f}, MAP: {:.4f}'.format(e, train_err, test_err, roc_auc_val, map_val)


if __name__ == '__main__':
    train, test = get_data('ml-100k')
    mf = ClassicMF(
        num_factors=50,
        learning_rate=0.07,
        optimizer=lasagne.updates.adagrad,
        regularization=0.0000003,
        num_epochs=500
    )
    mf.train(train, test)
