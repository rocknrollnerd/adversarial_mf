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


def get_data(dataset):
    if dataset == 'ml-100k':
        return MovieLens100k().get()
    elif dataset == '1m-songs':
        return KaggleMillionSongs().get()
    elif dataset == '1m-songs-subset':
        return pickle.load(open('datasets/1m-songs-subset.pkl'))
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
