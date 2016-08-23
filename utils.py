import numpy as np
import requests
import zipfile
from collections import defaultdict
from collections import Counter
from scipy import sparse
import cPickle as pickle
import pandas as pd
import os


def bincount_relative(arr, max_value, min_value):
    if arr.max() < max_value:
        # add max item index, so that bincount would work
        arr = np.append(arr, max_value)
    if arr.min() > min_value:
        arr = np.append(arr, min_value)
    counts = np.float32(np.bincount(arr))
    return counts / counts.sum()


class MovieLens100k(object):
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    dir_path = 'datasets/ml-100k'
    num_users = 943
    num_items = 1682

    def ensure_dir(self):
        if not os.path.exists('datasets'):
            os.makedirs('datasets')

    def already_exists(self):
        return all((
            os.path.exists(os.path.join(self.dir_path, 'ua.base')),
            os.path.exists(os.path.join(self.dir_path, 'ua.test'))
        ))

    def download(self):
        if self.already_exists():
            return
        self.ensure_dir()
        r = requests.get(self.url, stream=True)
        zip_path = os.path.join('/tmp', 'ml-100k.zip')

        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        zipped = zipfile.ZipFile(zip_path, 'r')
        zipped.extractall('datasets')
        zipped.close()

        try:
            os.remove(zip_path)
        except OSError:
            pass

    def get_train_test(self):
        self.download()

        train = np.zeros((self.num_users, self.num_items))
        test = np.zeros((self.num_users, self.num_items))

        for path, array in (('ua.base', train), ('ua.test', test)):
            with open(os.path.join(self.dir_path, path)) as f:
                for line in f:
                    user_id, item_id, rating, _ = map(int, line.strip().split('\t'))
                    array[user_id - 1, item_id - 1] = rating
        return np.float32(train), np.float32(test)

    def get(self):
        self.download()
        data = np.zeros((self.num_users, self.num_items))
        with open(os.path.join(self.dir_path, 'u.data')) as f:
            for line in f:
                user_id, item_id, rating, _ = map(int, line.strip().split('\t'))
                data[user_id - 1, item_id - 1] = rating
        return sparse.csr_matrix(np.float32(data))


class KaggleMillionSongs(object):

    def get_user_map(self, data):
        users = list(sorted(data.user.unique()))
        return dict(zip(users, range(len(users))))

    def get_item_map(self, data):
        items = list(sorted(data.item.unique()))
        return dict(zip(items, range(len(items))))

    def convert_to_matrix(self, data, user_map, item_map):
        users = []
        items = []
        ratings = []
        for row in data.itertuples():
            users.append(user_map[row.user])
            items.append(item_map[row.item])
            ratings.append(row.value)
        users = np.int32(users)
        items = np.int32(items)
        ratings = np.float32(ratings)

        return sparse.csr_matrix(
            (ratings, (users, items)), shape=(len(user_map), len(item_map))
        )

    def get(self, n=None):
        # 1450933
        df = pd.read_csv(
            'datasets/1m-songs/kaggle_visible_evaluation_triplets.txt',
            delimiter='\t', header=None, names=['user', 'item', 'value']
        )
        if n:
            df = df.sample(n)
        user_map = self.get_user_map(df)
        item_map = self.get_item_map(df)
        return self.convert_to_matrix(df, user_map, item_map)


class OnlineRetail(object):

    def get(self):
        cached_path = 'datasets/online_retail/cached.pkl'
        if os.path.exists(cached_path):
            return pickle.load(open(cached_path)).astype('float32')
        df = pd.read_excel('datasets/online_retail/Online Retail.xlsx')
        df = df.loc[pd.isnull(df.CustomerID) == False]
        df['CustomerID'] = df.CustomerID.astype(int)  # Convert to int for customer ID
        df = df[['StockCode', 'Quantity', 'CustomerID']]  # Get rid of unnecessary info
        df = df.groupby(['CustomerID', 'StockCode']).sum().reset_index()  # Group together
        df.Quantity.loc[df.Quantity == 0] = 1  # Replace a sum of zero purchases with a one to
        # indicate purchased
        df = df.query('Quantity > 0')  # Only get customers where purchase totals were positive

        customers = list(np.sort(df.CustomerID.unique()))  # Get our unique customers
        products = list(df.StockCode.unique())  # Get our unique products that were purchased
        quantity = list(df.Quantity)  # All of our purchases

        rows = df.CustomerID.astype('category', categories=customers).cat.codes
        # Get the associated row indices
        cols = df.StockCode.astype('category', categories=products).cat.codes
        # Get the associated column indices
        M = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
        pickle.dump(M, open(cached_path, 'w'))
        return M.astype('float32')


def get_data(dataset):
    if dataset == 'ml-100k':
        return MovieLens100k().get()
    elif dataset == '1m-songs':
        return KaggleMillionSongs().get()
    elif dataset == '1m-songs-subset':
        return pickle.load(open('datasets/1m-songs-subset.pkl'))
    elif dataset == 'online-retail':
        return OnlineRetail().get()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_index_pairs(arr):
    diffs = np.ediff1d(arr, to_begin=1, to_end=1)
    indices = diffs.nonzero()[0]
    return np.c_[indices[:-1], indices[1:]]


def sort_by_users(X, return_index=False):
    # sorting X by users
    idx = X[:, 0].argsort()
    X = X[idx]

    # split by user-continuous chunks
    index_pairs = get_index_pairs(X[:, 0])
    if not return_index:
        return X, index_pairs
    else:
        return X, index_pairs, idx


def add_implicit_negatives(X, n_items, factor=1):
    X_sorted, index_pairs = sort_by_users(X)
    n_samples = X_sorted.shape[0] * factor
    # make a user-by-items dictionary
    values = np.vstack([X_sorted[:, 0], X_sorted[:, 1]]).T
    data_dict = {}
    for pair in index_pairs:
        i, j = pair[0], pair[1]
        data_dict[values[i, 0]] = set(values[i: j, 1])

    sampled_users = np.repeat(values[:, 0], factor)
    left_out_users = np.arange(n_samples)
    sampled_negative_items = np.zeros(n_samples)

    # weight items according to popularity
    weights = bincount_relative(np.int32(values[:, 1]), max_value=n_items - 1, min_value=0)

    # sample negative items by sampling all items at once
    # and then retrying for items that are in `data_dict`
    while len(left_out_users) > 0:
        failed = []
        neg_items = np.random.choice(n_items, p=weights, size=len(left_out_users))
        for item_idx, user_idx in enumerate(left_out_users):
            user = sampled_users[user_idx]
            item = neg_items[item_idx]
            if item in data_dict[user]:
                failed.append(user_idx)
            else:
                sampled_negative_items[user_idx] = item
        left_out_users = failed

    negatives = np.vstack([
        sampled_users,
        sampled_negative_items,
        np.zeros(n_samples)
    ]).T

    # uncomment to check that sampling works correctly
    # check = dict(zip(zip(X_sorted[:, 0], X_sorted[:, 1]), X_sorted[:, 2]))
    # for i in negatives:
    #     if (i[0], i[1]) in check:
    #         print 'wrong sample', (i[0], i[1]), check[(i[0], i[1])]
    #         raise Exception('wrong sample')

    return np.vstack([X_sorted, negatives])
