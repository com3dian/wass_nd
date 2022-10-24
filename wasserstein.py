import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
import time

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
def wass_nd(sample_u, sample_v):
    '''
    n dimensional wasserstein function
    '''
    m, n = len(sample_u), len(sample_v)
    
    A_r = np.zeros((n, m, n))
    A_t = np.zeros((m, m, n))
    for i in range(n):
        for j in range(m):
            A_r[i, j, i] = 1
            A_t[j, j, i] = 1
    
    A = np.concatenate((A_t.reshape(m, m*n),
                        A_r.reshape(n, m*n)),
                        axis = 0)
    D = scipy.spatial.distance_matrix(sample_u, sample_v, p = 2)
    c = D.reshape((n*m))
    p_u, p_v = [1/m for i in range(m)], [1/n for i in range(n)]
    b = np.concatenate((p_u, p_v), axis = 0)
    opt_res = linprog( -b , A_ub = A.T, b_ub = c, bounds=(None, None))
    ans = b @ opt_res.x
    print('LP answer:', ans)
    return ans

@time_it
def scipy_wass(sample_u, sample_v):
    ans = wasserstein_distance(sample_u.flatten(), sample_v.flatten())
    print('scipy answer:', ans)
    return ans