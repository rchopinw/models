import numpy as np


class Bandit:
    def __init__(self, n, prob, seed=42):
        """
        :param n: total number of bandits
        :param prob: list of win rates
        :param seed: random seed, default to 42
        """
        assert n == len(prob), "Length of probabilities should be matched with n."
        self.n = n
        self.prob = prob
        self.rnd = np.random.RandomState(seed)

    def play(self, i):
        return 1 if self.rnd.random() < self.prob[i] else 0


class ThompsonSampling:
    def __init__(self, n, play_func, seed=42):
        """
        :param n: total number of bandits
        :param play_func: play one of the bandit by play_func(i)
        :param seed: random seed, default to 42
        """
        self.n = n
        self.pf = play_func
        self.random_seed = seed
        self.s = np.zeros(shape=self.n, dtype=np.int)
        self.f = np.zeros(shape=self.n, dtype=np.int)
        self.prob = np.zeros(shape=self.n, dtype=np.float)
        self.rnd = np.random.RandomState(self.random_seed)

    def sampling(self, max_iter=2000):
        for trial in range(max_iter):
            for i in range(self.n):
                self.prob[i] = self.rnd.beta(self.s[i] + 1, self.f[i] + 1)
            cur_optimal = self.prob.argmax()
            outcome = self.pf(cur_optimal)
            if outcome:
                self.s[cur_optimal] += 1
            else:
                self.f[cur_optimal] += 1
        return {'prob': self.prob, 'trial_success': self.s, 'trial_fail': self.f}


class BetaSampler:
    def __init__(self, seed=42):
        self.rnd = np.random.RandomState(seed)

    def next_sample(self, a, b):
        alpha = a + b
        beta, u1, u2, w, v = 0.0, 0.0, 0.0, 0.0, 0.0
        if min(a, b) <= 1.0:
            beta = max(1 / a, 1 / b)
        else:
            beta = np.sqrt((alpha - 2.0) / (2 * a * b - alpha))
        gamma = a + 1 / beta
        while True:
            u1 = self.rnd.random_sample()
            u2 = self.rnd.random_sample()
            v = beta * np.log(u1 / (1 - u1))
            w = a * np.exp(v)
            tmp = np.log(alpha / (b + w))
            if alpha * tmp + (gamma * v) - 1.3862944 >= np.log(u1 * u1 * u2):
                break
        x = w / (b + w)
        return x


if __name__ == '__main__':
    n = 10
    sample_iter = 2000
    prob = [0.1, 0.3, 0.2, 0.8, 0.5, 0.4, 0.3, 0.7, 0.44, 0.66]
    bandits = Bandit(n, prob)
    ts = ThompsonSampling(n, bandits.play)
    res = ts.sampling(sample_iter)
    print('After {} iterations, the sampled probabilities are: {}.'.format(sample_iter, res['prob']))