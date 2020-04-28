import numpy as np
from scipy import stats

def mode(y):
    if len(y) == 0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def euclidean_dist_squared(X, Xtest):
    return np.sum(X ** 2, axis=1)[:, None] + np.sum(Xtest ** 2, axis=1)[None] - 2 * np.dot(X, Xtest.T)

def classification_error(y, yhat):
    return np.mean(y != yhat)


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm

def sigmoid(z):
    sig = 1 / (1-np.exp(-z))
    return sig