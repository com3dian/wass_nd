import time
import numpy as np
from scipy.stats import wasserstein_distance


def mixedGaussian(mu1, mu2, sigma1, sigma2, n):
    bernoulli = np.random.binomial(n = 1, p = 0.5, size = n)
    gaussian1 = np.random.normal(mu1, sigma1, n)
    gaussian2 = np.random.normal(mu2, sigma2, n)
    return (gaussian1**bernoulli)*(gaussian2**(1-bernoulli))


def time_it(func):
    def inner(*args, **kw):
        start = time.time()
        func(*args, **kw)
        end = time.time()
        print('time:{}seconds'.format(end-start))
    return inner

@time_it
def scipy_wass(sample_u, sample_v):
    ans = wasserstein_distance(sample_u.flatten(), sample_v.flatten())
    print('scipy answer:', ans)
    return ans