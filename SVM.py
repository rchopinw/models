import cvxopt
import cvxopt.solvers
import numpy as np


class SVM:
    def __init__(self, c, kernel='linear'):
        self.c = c
        self.kernel = kernel
        if self.kernel == 'linear':
            self.kernel_f = self.linear_kernel
        elif self.kernel == 'polynomial':
            self.kernel_f = self.polynomial_kernel
        else:
            self.kernel_f = self.rbf_kernel

    @staticmethod
    def linear_kernel(x, y):
        return x @ y

    @staticmethod
    def polynomial_kernel(x, y, p=3):
        return (1 + x @ y) ** p

    @staticmethod
    def rbf_kernel(x, y, sigma=5.0):
        return np.exp(-np.linalg.norm(x - y)**2 / 2 * (sigma ** 2))

    def fit(self, x, y):
        n, d = x.shape
        kernel = np.zeros(shape=(n, d))
        for i in range(n):
            for j in range(d):
                kernel[i, j] = self.kernel_f(x[i], x[j])
        # solving:
        """
        MIN 1/2 * sum_{i}sum_{j}(alpha_j * alpha_k * y_j * y_k * Kernel(x_i, x_j)) - sum_{i}alpha_i
        subject to: 0 <= alpha_i < C, sum_{i}alpha_i * y_i = 0
        """
        p = cvxopt.matrix(np.outer(y, y) * kernel)
        q = cvxopt.matrix(np.ones(shape=n) * -1)
        a = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(0.0)
        if self.c is None:
            g = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            tmp1 = np.diag(np.ones(n) * -1)
            tmp2 = np.identity(n)
            g = cvxopt.matrix(np.vstack([tmp1, tmp2]))
            tmp3 = np.zeros(shape=n)
            tmp4 = np.ones(n) * self.c
            h = cvxopt.matrix(np.hstack([tmp3, tmp4]))
        sol = cvxopt.solvers.qp(p, q, g, h, a, b)
        a = np.ravel(sol['x'])
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = x[sv]
        self.sv_y = y[sv]
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * kernel[ind[n], sv])
        self.b /= len(self.a)
        if self.kernel == 'linear':
            self.w = np.zeros(n)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
