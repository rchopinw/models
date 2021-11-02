import numpy as np
import heapq
from collections import Counter


class KNN:
    def __init__(self, k, x, y):
        self.k = k
        self.x = x
        self.n, self.d = x.shape
        self.y = y

    def predict(self, x):
        pq = []
        for obs, c in zip(self.x, self.y):
            distance = np.linalg.norm(obs - x)
            if len(pq) >= self.k:
                if distance > pq[0]:
                    heapq.heappushpop(pq, (distance, c))
            else:
                heapq.heappush(pq, (distance, c))
        count = Counter(pq)
        return count.most_common(1)[0][0]
