import numpy as np
import optimization


class multiSVM:
    def __init__(self, lammy, epochs, alpha, batch):
        self.lammy = lammy
        self.epochs = epochs
        self.alpha = alpha
        self.batch = batch

    def funObj(self, w, X, y):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.k = np.unique(y).size
        f = 0
        g = np.zeros([self.k, self.d])
        for i in range(self.n):
            scores = np.dot(X[i], w.T)  # scores = w^Tx_i
            label = scores[int(y[i])]  # labels = y_hat
            for c in range(self.k):
                margin = 1 + scores[c] - label
                if y[i] != c and margin > 0:
                    f += margin
                    g[int(y[i]), :] -= X[i, :]
                    g[c, :] += X[i, :]
        f /= self.n
        f += self.lammy / 2 * np.square(np.linalg.norm(w))
        g /= self.n
        g += self.lammy * w
        return f, g

    def fit(self, X, y):
        k = np.unique(y).size
        self.w = np.zeros([k, X.shape[1]])
        self.w, f = optimization.sgd(self.funObj, self.w, X, y, self.epochs, self.batch, self.alpha)
        return self.w

    def predict(self, X):
        prob = np.dot(X, self.w.T)
        predict = np.argmax(prob, axis=1)
        return predict
