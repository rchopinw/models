import numpy as np


class LogisticRegression:
    """
    X: n x d (n observations, d features)
    W: d x k (d features, k classes)
    """
    def __init__(self, lam=0):
        self.c = lam

    def fit(self, x, y, lr=0.001, max_iter=10000):
        y = self._one_hot(y)
        x = np.hstack([np.ones(shape=(x.shape[0], 1)), x])
        self.w = np.random.random(size=(x.shape[1], len(self.label)))
        self.loss_history = []
        for i in range(max_iter):
            gradient, loss = self.__gradient(x, y, self.w), self.__loss(x, y, self.w)
            self.w = self.w - lr * gradient
            self.loss_history.append(loss)

    def predict(self, x):
        x = np.hstack(
            [np.ones(shape=(x.shape[0], 1)), x]
        )
        prob = self.__softmax(x @ self.w)
        arg_max = prob.max(axis=1)
        return np.array(
            [
                [self.label_inverse[x]]
                for x in arg_max
            ]
        )

    def __softmax(self, x):
        x_exp = np.exp(x)
        norm = x_exp.sum(axis=1, keepdims=True)
        return x_exp / norm

    def __gradient(self, x, y, w):
        g = - x.T @ (y - self.__softmax(x @ w)) / x.shape[0] + self.c * w
        return g

    def __loss(self, x, y, w):
        l = - np.sum(y * np.log(self.__softmax(x @ w))) / x.shape[0] + (self.c / 2) * np.sum(w * w)
        return l

    def _one_hot(self, y):
        y_unique = np.sort(np.unique(y))
        self.label = {x: i for i, x in enumerate(y_unique)}
        self.label_inverse = {self.label[x]: x for x in self.label}
        y_one_hot = np.zeros(shape=(len(y), len(y_unique)))
        for i, c in enumerate(y.flatten()):
            y_one_hot[i, self.label[c]] = 1
        return y_one_hot


class BinaryLogisticRegression:
    def __init__(self, lam=0):
        self.c = lam

    def fit(self, x, y, lr=0.0001, max_iter=2000):
        x = np.hstack(
            [np.ones(shape=(x.shape[0], 1)), x]
        )
        self.w = np.random.random(size=(x.shape[0], 1))
        self.loss = []
        for i in range(max_iter):
            z = x @ self.w
            h = self._sigmoid(z)
            g = x.T @ (h - y) / y.shape[0]
            self.w = self.w - lr * g
            self.loss.append(self._loss(h, y))

    def _predict_prob(self, x):
        x = np.hstack(
            [np.ones(shape=(x.shape[0], 1)), x.reshape((-1, 1))]
        )
        return self._sigmoid(x)

    def _predict(self, x, threshold=0.5):
        return self._predict_prob(x) >= threshold

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _loss(self, x, y):
        return np.mean(-y * np.log(x) - (1 - y) * np.log(1 - x))
