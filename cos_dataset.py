#!/usr/bin/env python
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import scale


def noise(t, n):
    return {'gaussian': np.random.randn,
            'uniform': lambda x: np.random.uniform(0.2,
                                                   np.random.uniform(0.5,1),
                                                   x)}[t](n)


def gmm_cause(points, k=2, p1=3, p2=4):
    """Init a root cause with a Gaussian Mixture Model w/ a spherical covariance type."""
    g = GMM(k, covariance_type="spherical")
    g.fit(np.random.randn(300, 1))

    g.means_ = p1 * np.random.randn(k, 1)
    g.covars_ = np.power(abs(p2 * np.random.randn(k, 1) + 1), 2)
    g.weights_ = abs(np.random.rand(k))
    g.weights_ = g.weights_ / sum(g.weights_)
    return g.sample(points)[0].reshape(-1)


def cause(t, n):
    return {'gmm': gmm_cause,
            'normal': np.random.randn}[t](n)


def generate_pair(n, max_w=1.2, max_phi=3.14,
                  noisef='gaussian', causef='gmm'):
    c = cause(causef, n)
    return scale(c), scale(np.sin(np.random.normal(0.2, max_w) * c
                                  + np.random.uniform(-max_phi, max_phi))
                           + noise(noisef, n))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    for i in range(1, 26):
        plt.subplot(5, 5, i)
        plt.xticks(())
        plt.yticks(())
        pair = generate_pair(500, noisef=np.random.choice(['uniform', 'gaussian']),
                                   causef=np.random.choice(['gmm', 'normal']))
        pair = pair[::-1] if np.random.choice([True, False]) else pair
        plt.scatter(*pair, marker=".")
        print(i)
    plt.show()
