import os
import pickle
import gzip
import argparse
import numpy as np
import math
from scipy import stats

from sklearn.preprocessing import LabelBinarizer

from knn import KNN
import utils
import linear_model
import svm
import mlp


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

    elif question == "knn":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        N, D1 = X.shape  # X is a N by D1 matrix
        T, D2 = Xtest.shape  # Xtest is a T by D2 matrix
        K = int(math.log(N, 2))  # K is the hyper parameter for KNN algorithm
        print("N = ", N)
        print("T = ", T)
        print("K = ", K)

        model = KNN(K)
        model.fit(X, y)

        # Because the data is big, iterate through 1000 examples at a time and compute the errors for each partitions.
        # Then compute the overall averages of the errors to get the error.
        i = 0
        k = 0
        training_error = []
        test_error = []
        while i < N:
            print(i)
            j = i + 1000
            temp = X[i:j]
            pred = model.predict(temp)
            temp_training_error = np.mean(pred != y[i:j])
            training_error = np.append(training_error, temp_training_error)
            i = j
        while k < T:
            print(k)
            l = k + 1000
            temp = Xtest[k:l]
            pred = model.predict(temp)
            temp_test_error = np.mean(pred != ytest[k:l])
            test_error = np.append(test_error, temp_test_error)
            k = l

        print("training error", np.mean(training_error), "test error", np.mean(test_error))

    elif question == "knn2":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        N, D1 = X.shape  # X is a N by D1 matrix
        T, D2 = Xtest.shape  # Xtest is a T by D2 matrix
        print("N = ", N)
        print("T = ", T)

        K = 1
        while K < 4:
            model = KNN(K)
            model.fit(X, y)

            # Because the data is big, iterate through 1000 examples at a time and compute the errors for each partitions.
            # Then compute the overall averages of the errors to get the error.
            i = 0
            k = 0
            training_error = []
            test_error = []
            while i < N:
                print(i)
                j = i + 1000
                temp = X[i:j]
                pred = model.predict(temp)
                temp_training_error = np.mean(pred != y[i:j])
                training_error = np.append(training_error, temp_training_error)
                i = j
            while k < T:
                print(k)
                l = k + 1000
                temp = Xtest[k:l]
                pred = model.predict(temp)
                temp_test_error = np.mean(pred != ytest[k:l])
                test_error = np.append(test_error, temp_test_error)
                k = l
            print("for K = ", K, "training error", np.mean(training_error), "test error", np.mean(test_error))
            K = K + 1

    elif question == "linear":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        # print(y)
        X = np.float32(X)
        y = np.float32(y)
        Xtest = np.float32(Xtest)
        ytest = np.float32(ytest)

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)
        y_int = np.int32(y)
        # Ytest = binarizer.fit_transform(ytest)

        model = linear_model.softmaxClassifier(lammy=1, maxEvals=10, alpha=1e-2, batch=5000)

        model.fit(X, y, Y)

        pred = model.predict(Xtest)
        print("test error is", utils.classification_error(ytest, pred))

    elif question == "svm":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        X = np.float32(X)
        y = np.float32(y)
        Xtest = np.float32(Xtest)
        ytest = np.float32(ytest)

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        model = svm.multiSVM(lammy=1, maxEvals=10, alpha=1e-2, batch=5000)

        W = model.fit(X, y)

        print(W)

        pred = model.predict(Xtest)

        print("test error is", utils.classification_error(ytest, pred))


    elif question == "mlp":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        X = np.float32(X)
        y = np.float32(y)
        Xtest = np.float32(Xtest)
        ytest = np.float32(ytest)

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        model = mlp.NeuralNet(hidden_layer_sizes=[500], lammy=1, max_iter=500)
        model.fit(X, y)
        pred = model.predict(Xtest)
        print("test error is", utils.classification_error(ytest, pred))

    elif question == "cnn":
        print("cnn")

    else:
        print("Unknown question: %s" % question)
