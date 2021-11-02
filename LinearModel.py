import numpy as np


class LinearRegression:

    def __init__(self, regularization=None, c=0):
        self.regularization = regularization
        self.c = c

    def fit(self, x: np.array, y: np.array):
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        if self.regularization == 'l1':
            part1 = np.linalg.solve(x.T @ x, x.T @ y)
            part2 = np.abs(part1) - self.c
            part2[part2 < 0] = 0
            self.w = np.sign(part1) * part2
        elif self.regularization == 'l2':
            self.w = np.linalg.solve(x.T @ x + self.c * np.eye(x.shape[0]), x.T @ y)
        else:
            self.w = np.linalg.solve(x.T @ x, x.T @ y)

    def predict(self, x):
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        return x @ self.w
