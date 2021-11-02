import numpy as np


class Optimizer:
    def __init__(self, func):
        self.grad_func = func

    def grad(self, w, max_iter=2000, lr=0.00001):
        for _ in range(max_iter):
            g = self.grad_func(w)
            w = w - lr * g
        return w

    def grad_momentum(self, w, lr=0.0001, mu=0.99, max_iter=2000):
        m = np.zeros(shape=w.shape)
        for _ in range(max_iter):
            g = self.grad_func(w)
            m = mu * m + g
            w = w - lr * m
        return w

    def rms_prop(self, w, lr=0.0001, v=0.0001, max_iter=2000, epsilon=0.00001):
        n = np.zeros(shape=w.shape)
        for _ in range(max_iter):
            g = self.grad_func(w)
            n = v * n + (1 - v) * g**2
            w = w - lr * g / (np.sqrt(n) + epsilon)
        return w

    def ada_grad(self, w, lr=0.0001, max_iter=2000, epsilon=0.00001):
        n = np.zeros(shape=w.shape)
        for _ in range(max_iter):
            g = self.grad_func(w)
            n = n + g**2
            w = w - lr * g / (np.sqrt(n) + epsilon)
        return w

    def adam(self, w, lr=0.0001, mu=0.99, v=0.999, max_iter=2000, epsilon=0.00001):
        m, n = np.zeros(shape=w.shape), np.zeros(shape=w.shape)
        for i in range(1, max_iter+1):
            g = self.grad_func(w)
            m = mu * m + (1 - mu) * g
            n = v * n + (1 - v) * g**2
            m = m / (1 - mu**i)
            n = n / (1 - v**i)
            w = w - lr * m / (np.sqrt(n) + epsilon)
        return w

    def ada_max(self, ):
        pass

    def ada_delta(self, ):
        pass

    def n_adam(self, ):
        pass

    def adam_w(self, w, lr=0.0001, mu=0.99, v=0.999, max_iter=2000, epsilon=0.00001):
        m, n = np.zeros(shape=w.shape), np.zeros(shape=w.shape)
        for i in range(1, max_iter+1):
            g = self.grad_func(w)
            m = mu * m + (1 - mu) * g
            n = v * n + (1 - v) * g**2
            m = m / (1 - mu**i)
            n = n / (1 - v**i)
            w = w - lr * m / (np.sqrt(n) + epsilon) - lr * w * g
        return w

    def nag(self, ):
        pass
