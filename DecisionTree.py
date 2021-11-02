import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, func='entropy', task='classification'):
        if func == 'impurity':
            self.func = self.impurity
        else:
            self.func = self.entropy
        self.task = task

    def information_gain(self, y, mask):
        a = sum(mask)
        b = mask.shape[0] - a
        if a == 0 or b == 0:
            ig = 0
        else:
            if self.task == 'classification':
                ig = self.func(y) - a / (a + b) * self.func(y[mask]) - b / (a + b) * self.func(y[-mask])
            else:
                ig = self.variance(y) - a / (a + b) * self.variance(y[mask]) - b / (a + b) * self.variance(y[-mask])
        return ig

    @staticmethod
    def variance(y):
        if len(y) == 1:
            return 0
        else:
            return np.var(y)

    @staticmethod
    def entropy(y):
        a = np.array(Counter(y).values()) / y.shape[0]
        e = np.sum(-a * np.log(a + 1e-9))
        return e

    @staticmethod
    def impurity(y):
        p = np.array(Counter(y).values()) / y.shape[0]
        gini = 1 - np.sum(p ** 2)
        return gini

