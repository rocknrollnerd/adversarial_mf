import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.cross_validation import train_test_split

from utils import get_data
from metrics import *

from base import BaseRecommender


class AdversarialMF(BaseRecommender):
    default_discriminator_params = (
        {'n_neurons': 200, 'nonlinearity': leaky_rectify},
        {'n_neurons': 100, 'nonlinearity': leaky_rectify},
        {'n_neurons': 1, 'nonlinearity': sigmoid}
    )

    def __init__(self, n_users, n_items, n_factors=20, g_lr=0.001, d_lr=0.001, optimizer=lasagne.updates.adagrad, discriminator_k=3, n_epochs=100, discriminator_regularization=0., generator_regularization=0., discriminator_params=None):
        super(AdversarialMF, self).__init__(n_users, n_items)
        self.n_factors = n_factors
        self.discriminator_regularization = discriminator_regularization
        self.generator_regularization = generator_regularization
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.discriminator_k = discriminator_k
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.discriminator_params = discriminator_params or self.default_discriminator_params

    def initialize(self, input_data):
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

        # building discriminator
        d1 = lasagne.layers.InputLayer((None, len(input_data)), ratings.reshape([1, -1]))
        d2 = lasagne.layers.InputLayer((None, len(input_data)), predictions.reshape([1, -1]))
        for layer_params in self.discriminator_params:
            d1 = lasagne.layers.DenseLayer(d1, layer_params['n_neurons'], nonlinearity=layer_params['nonlinearity'])
            d2 = lasagne.layers.DenseLayer(d2, layer_params['n_neurons'], nonlinearity=layer_params['nonlinearity'], W=d1.W, b=d1.b)

        # outputs and objectives
        d1_out = lasagne.layers.get_output(d1)
        d2_out = lasagne.layers.get_output(d2)
        d_obj = (T.log(d1_out) + T.log(1 - d2_out)).mean()
        d_params = lasagne.layers.get_all_params(d1, trainable=True)
        d_lr = theano.shared(np.array(self.d_lr, dtype=theano.config.floatX))
        d_loss = (1 - d_obj) + self.discriminator_regularization * lasagne.regularization.regularize_network_params(d1, lasagne.regularization.l2)
        d_updates = self.optimizer(d_loss, d_params, learning_rate=d_lr)
        self.d_train = theano.function([users, items, ratings], d_obj, updates=d_updates)

        # outputs and objectives for generator/approximator
        g_obj = T.log(d2_out).mean()
        g_params = [self.U, self.I]
        g_lr = theano.shared(np.array(self.g_lr, dtype=theano.config.floatX))
        g_loss = (1 - g_obj) + self.generator_regularization * (T.sum(self.U ** 2) + T.sum(self.I ** 2))
        g_updates = self.optimizer(g_loss, g_params, learning_rate=g_lr)
        self.g_train = theano.function([users, items], g_obj, updates=g_updates)

        self.predict_theano = theano.function([users, items], predictions)

    def fit(self, X, y=None):
        X_train, X_test = train_test_split(X, test_size=0.1)
        train_users, train_items, train_ratings = np.int32(X_train[:, 0]), np.int32(X_train[:, 1]), X_train[:, 2]
        # test_users, test_items, test_ratings = np.int32(X_test[:, 0]), np.int32(X_test[:, 1]), X_test[:, 2]
        self.initialize(train_ratings)
        self.popularity_ = self.count_popularity(train_items)

        for e in xrange(self.n_epochs):
            g_values = []
            d_values = []
            # train discriminator
            for _ in xrange(self.discriminator_k):
                d_obj = self.d_train(train_users, train_items, train_ratings)
            d_values.append(d_obj)
            # train generator
            g_obj = self.g_train(train_users, train_items)
            g_values.append(g_obj)
            print e, avg_roc_sauc_weighted(self, X_test), g_obj, d_obj
            # print e, g_obj, d_obj

        return self

    def predict(self, users, items):
        return self.predict_theano(np.int32(users), np.int32(items))
