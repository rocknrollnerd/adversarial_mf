from sklearn import metrics
import matplotlib.pyplot as plt

from utils import *


def binarize_ratings(ratings, threshold):
    ratings_binary = ratings.copy()
    ratings_binary[ratings_binary < threshold] = 0
    ratings_binary[ratings_binary >= threshold] = 1
    return ratings_binary


def mean_average_precision(test_ratings, test_predictions, threshold=4):
    test_ratings = binarize_ratings(test_ratings, threshold)
    return metrics.average_precision_score(test_ratings, test_predictions)


def pr_auc(test_ratings, test_predictions, threshold=4, visualize=False):
    # the same as mean_average_precision
    test_ratings = binarize_ratings(test_ratings, threshold)
    precision, recall, thresholds = metrics.precision_recall_curve(test_ratings, test_predictions)
    auc = metrics.auc(recall, precision)
    if visualize:
        print 'AUC', auc
        plt.plot(recall, precision)
        plt.show()
    return auc


def roc_auc(test_ratings, test_predictions, threshold=4, visualize=False):
    test_ratings = binarize_ratings(test_ratings, threshold)
    fpr, tpr, thresholds = metrics.roc_curve(test_ratings, test_predictions)
    auc = metrics.auc(fpr, tpr)
    if visualize:
        print 'AUC', auc
        plt.plot(fpr, tpr)
        plt.show()
    return auc
