import numpy as np
from collections import defaultdict


class KMean:
    def __init__(self, k, tol=1e-5, max_iter=2000, distance='euclidean'):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        if distance == 'cosine':
            self.dist = self.cosine_similarity
        elif distance == 'manhattan':
            self.dist = self.manhattan_distance
        else:
            self.dist = self.eculidean_distance

    def fit(self, x, cal_loss=False):
        self.centroids = {}
        self.loss_within_class = defaultdict(list)

        for c in range(self.k):
            self.centroids[c] = x[c]

        for i in range(self.max_iter):
            self.classifications = {}

            for c in range(self.k):
                self.classifications[c] = []

            for obs in x:
                distances = [self.dist(obs, self.centroids[c]) for c in self.classifications]
                cur_class = distances.index(max(distances))
                self.classifications[cur_class].append(obs)

            for c in range(self.k):
                self.centroids[c] = np.mean(self.classifications[c], axis=0)

            if cal_loss:
                for c in range(self.k):
                    loss = sum(self.dist(self.centroids[c], obs) for obs in self.classifications[c])
                    self.loss_within_class[i].append(loss/len(self.classifications[c]))

    def predict(self, x):
        distances = [self.dist(x, self.centroids[c]) for c in self.centroids]
        return distances.index(min(distances))

    @staticmethod
    def eculidean_distance(x, y):
        return np.linalg.norm(x - y, ord=2)

    @staticmethod
    def manhattan_distance(x, y):
        return np.linalg.norm(x - y, ord=1)

    @staticmethod
    def cosine_similarity(x, y):
        return np.inner(x, y) / (np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))

