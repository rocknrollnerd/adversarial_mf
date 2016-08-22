# from sklearn import metrics
# import matplotlib.pyplot as plt

# from utils import *
from collections import defaultdict
import numpy as np


def group_by_users(users, items, ratings):
    result = defaultdict(list)
    for i in xrange(len(users)):
        result[users[i]].append((items[i], ratings[i]))
    return result


def avg_roc_auc(recommender, X, y=None):
    users, items, ratings = np.int32(X[:, 0]), np.int32(X[:, 1]), X[:, 2]
    scores = recommender.predict(users, items)

    rating_dict = group_by_users(users, items, ratings)
    scores_dict = group_by_users(users, items, scores)
    auc_values = []
    for user in rating_dict:
        user_items, user_ratings = zip(*rating_dict[user])
        _, user_scores = zip(*scores_dict[user])
        hits = 0
        total = 0
        for i in xrange(len(user_ratings)):
            for j in xrange(len(user_scores)):
                if user_ratings[i] == user_ratings[j]:
                    continue
                total += 1
                if (user_ratings[i] > user_ratings[j]) and (user_scores[i] > user_scores[j]):
                    hits += 1
                elif (user_ratings[i] < user_ratings[j]) and (user_scores[i] < user_scores[j]):
                    hits += 1
        if not hits:
            auc = 0.5
        else:
            auc = hits / float(total)
        auc_values.append(auc)
    return np.mean(auc_values)


def avg_roc_sauc_binary(recommender, X, y=None):
    users, items, ratings = np.int32(X[:, 0]), np.int32(X[:, 1]), X[:, 2]
    scores = recommender.predict(users, items)
    popularity = recommender.popularity_

    rating_dict = group_by_users(users, items, ratings)
    scores_dict = group_by_users(users, items, scores)
    auc_values = []
    for user in rating_dict:
        user_items, user_ratings = zip(*rating_dict[user])
        _, user_scores = zip(*scores_dict[user])
        hits = 0
        total = 0
        for i in xrange(len(user_ratings)):
            for j in xrange(len(user_scores)):
                if user_ratings[i] == user_ratings[j]:
                    continue
                if (user_ratings[i] > user_ratings[j]):
                    if popularity[user_items[i]] < popularity[user_items[j]]:
                        total += 1
                        if (user_scores[i] > user_scores[j]):
                            hits += 1
                elif (user_ratings[i] < user_ratings[j]):
                    if popularity[user_items[i]] > popularity[user_items[j]]:
                        total += 1
                        if (user_scores[i] < user_scores[j]):
                            hits += 1
        # print hits, total
        if not total:
            continue
        if not hits:
            auc = 0.5
        else:
            auc = hits / float(total)
        auc_values.append(auc)
    return np.mean(auc_values)


def avg_roc_sauc_weighted(recommender, X, y=None):
    users, items, ratings = np.int32(X[:, 0]), np.int32(X[:, 1]), X[:, 2]
    scores = recommender.predict(users, items)
    popularity = recommender.popularity_

    rating_dict = group_by_users(users, items, ratings)
    scores_dict = group_by_users(users, items, scores)
    auc_values = []
    for user in rating_dict:
        user_items, user_ratings = zip(*rating_dict[user])
        _, user_scores = zip(*scores_dict[user])
        hits = 0
        total = 0
        for i in xrange(len(user_ratings)):
            for j in xrange(len(user_scores)):
                if user_ratings[i] == user_ratings[j]:
                    continue
                if (user_ratings[i] > user_ratings[j]):
                    total += popularity[j]
                    if (user_scores[i] > user_scores[j]):
                        hits += popularity[j]
                elif (user_ratings[i] < user_ratings[j]):
                    total += popularity[i]
                    if (user_scores[i] < user_scores[j]):
                        hits += popularity[i]
        if not total:
            continue
        if not hits:
            auc = 0.5
        else:
            auc = hits / float(total)
        auc_values.append(auc)
    return np.mean(auc_values)
