import numpy as np
import requests
import zipfile
from collections import defaultdict
from collections import Counter
from scipy import sparse
import os


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

    def get(self):
        self.download()

        train = np.zeros((self.num_users, self.num_items))
        test = np.zeros((self.num_users, self.num_items))

        for path, array in (('ua.base', train), ('ua.test', test)):
            with open(os.path.join(self.dir_path, path)) as f:
                for line in f:
                    user_id, item_id, rating, _ = map(int, line.strip().split('\t'))
                    array[user_id - 1, item_id - 1] = rating

        return np.float32(train), np.float32(test)

    def get_base(self):
        self.download()
        data = np.zeros((self.num_users, self.num_items))
        with open(os.path.join(self.dir_path, 'u.data')) as f:
            for line in f:
                user_id, item_id, rating, _ = map(int, line.strip().split('\t'))
                data[user_id - 1, item_id - 1] = rating
        return np.float32(data)


def get_data(dataset):
    if dataset == 'ml-100k':
        return MovieLens100k().get()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
