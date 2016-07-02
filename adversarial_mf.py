import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from utils import get_data
from metrics import roc_auc, mean_average_precision


class AdversarialMF(object):

    def __init__(self, num_factors=20, n_discriminators=3, g_lr=0.01, d_lr=0.1, optimizer=lasagne.updates.adagrad, discriminator_k=10, num_epochs=100, discriminator_input_size=1000):
        self.num_factors = num_factors
        self.n_discriminators = n_discriminators
        self.discriminator_input_size = discriminator_input_size
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.discriminator_k = discriminator_k
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.indices = []
        self.d_train_functions = []
        self.g_train_functions = []

    def initialize(self, train, test, train_input_size):
        num_users, num_items = train.shape
        self.train_input_size = train_input_size

        # theano stuff
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

        predictions = (self.U[users] * self.I[items]).sum(axis=1).reshape([1, -1])
        self.predict = theano.function([users, items], predictions.ravel())

        # discriminators
        if self.discriminator_input_size:
            discriminator_idx_size = self.discriminator_input_size
        else:
            discriminator_idx_size = int(train_input_size / float(self.n_discriminators))

        for _ in xrange(self.n_discriminators):
            sample_idx = np.random.choice(train_input_size, discriminator_idx_size, replace=False)

            d_input = T.matrix()
            d1_l1 = lasagne.layers.InputLayer((None, discriminator_idx_size), d_input)
            d1_l2 = lasagne.layers.DenseLayer(d1_l1, 200, nonlinearity=rectify)
            d1_l3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(d1_l2, 0.5), 100, nonlinearity=rectify)
            d1_l4 = lasagne.layers.DenseLayer(lasagne.layers.dropout(d1_l3, 0.5), 1, nonlinearity=sigmoid)
            d1 = d1_l4

            d2_l1 = lasagne.layers.InputLayer((None, discriminator_idx_size), predictions[:, sample_idx])
            d2_l2 = lasagne.layers.DenseLayer(d2_l1, 200, nonlinearity=rectify, W=d1_l2.W, b=d1_l2.b)
            d2_l3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(d2_l2, 0.5), 100, nonlinearity=rectify, W=d1_l3.W, b=d1_l3.b)
            d2_l4 = lasagne.layers.DenseLayer(lasagne.layers.dropout(d2_l3, 0.5), 1, nonlinearity=sigmoid, W=d1_l4.W, b=d1_l4.b)
            d2 = d2_l4

            d1_out = lasagne.layers.get_output(d1)
            d2_out = lasagne.layers.get_output(d2)

            g_obj = (T.log(d2_out)).mean()
            d_obj = (T.log(d1_out) + T.log(1 - d2_out)).mean()

            g_params = [self.U, self.I]
            g_lr = theano.shared(np.array(self.g_lr, dtype=theano.config.floatX))
            g_updates = self.optimizer(1 - g_obj, g_params, learning_rate=g_lr)

            d_params = lasagne.layers.get_all_params(d1, trainable=True)
            d_lr = theano.shared(np.array(self.d_lr, dtype=theano.config.floatX))
            d_updates = self.optimizer(1 - d_obj, d_params, learning_rate=d_lr)

            self.indices.append(sample_idx)
            self.d_train_functions.append(
                theano.function([users, items, d_input], d_obj, updates=d_updates)
            )
            self.g_train_functions.append(
                theano.function([users, items], g_obj, updates=g_updates)
            )
        self.test_rmse = theano.function(
            [users, items, ratings],
            T.sqrt(((predictions - ratings) ** 2).mean())
        )

    def train(self, train, test):
        train_coo = sparse.coo_matrix(train)
        test_coo = sparse.coo_matrix(test)

        train_users = np.int32(train_coo.row)
        train_items = np.int32(train_coo.col)
        train_ratings = np.float32(train_coo.data)
        test_users = np.int32(test_coo.row)
        test_items = np.int32(test_coo.col)
        test_ratings = np.float32(test_coo.data)

        self.initialize(train, test, len(train_users))

        plt.ion()
        plt.figure(figsize=(25, 15))
        for e in xrange(self.num_epochs):

            g_errors = []
            d_errors = []
            for i in xrange(self.n_discriminators):
                idx = self.indices[i]
                d_train = self.d_train_functions[i]
                g_train = self.g_train_functions[i]

                sample_ratings = train_ratings[idx].reshape(1, -1)

                for j in xrange(self.discriminator_k):
                    d_obj = d_train(train_users, train_items, sample_ratings)
                g_obj = g_train(train_users, train_items)

                d_errors.append(d_obj)
                g_errors.append(g_obj)

            test_predictions = self.predict(test_users, test_items)
            roc_auc_val = roc_auc(test_ratings, test_predictions)
            map_val = mean_average_precision(test_ratings, test_predictions)

            print 'epoch {}, mean d error: {:.6f}, mean g error: {:.6f}, ROC AUC: {:.4f}, MAP: {:.4f}'.format(e, np.mean(d_errors), np.mean(g_errors), roc_auc_val, map_val)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(train, interpolation='none')
            plt.subplot(1, 2, 2)
            plt.imshow(self.U.get_value().dot(self.I.get_value().T), interpolation='none')
            plt.draw()
        plt.ioff()


if __name__ == '__main__':
    train, test = get_data('ml-100k')
    mf = AdversarialMF(
        num_factors=50,
        n_discriminators=3,
        g_lr=0.01,
        d_lr=0.1,
        optimizer=lasagne.updates.adagrad,
        discriminator_k=10,
        num_epochs=500,
        discriminator_input_size=None,
    )
    mf.train(train, test)
