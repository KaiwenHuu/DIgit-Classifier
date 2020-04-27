import numpy as np
import utils
import optimization


class softmaxClassifier:
    def __init__(self, lammy, epochs, alpha, batch):
        self.lammy = lammy
        self.epochs = epochs
        self.alpha = alpha
        self.batch = batch

    def funObj(self, w, X, Y):
        n = X.shape[0]
        scores = np.dot(X, w)
        prob = utils.softmax(scores)
        f = (-1 / n) * np.sum(Y * np.log(prob)) + (self.lammy / 2) * np.sum(w * w)
        g = (-1 / n) * np.dot(X.T, (Y - prob)) + self.lammy * w
        return f, g

    def fit(self, X, y, Y):
        k = np.unique(y).size
        self.w = np.zeros([X.shape[1], k])
        self.w, f = optimization.sgd(self.funObj, self.w, X, Y, self.epochs, self.batch, self.alpha)
        return self.w

    def predict(self, X):
        probs = utils.softmax(np.dot(X, self.w))
        pred = np.argmax(probs, axis=1)
        return pred
