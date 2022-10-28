import numpy as np
import scipy
from test_utils import *

# todo: fast sinkhorn - 1 algorithm
# https://arxiv.org/pdf/2202.10042.pdf

# todo: stocahstic large sinkhorn algorithm
# Stochastic Optimization for Large-scale Optimal Transport

# todo: probably can also consider stabilized sinkhorn ?->
# https://arxiv.org/pdf/1610.06519.pdf

# todo: maxiter paramter in sinkhorn?


@time_it
def wass_naive_sinkhorn(sample_u, sample_v, 
                        u_weight = None, v_weight = None, 
                        epsilon = 0.1, precision = 1e-7):
    '''
    A naive implementaion of sinkhorn algorithm
    based on lucy Liu's code -> https://lucyliu-ucsb.github.io/posts/Sinkhorn-algorithm/

    '''
    element_max = np.vectorize(max)
    
    m, n = len(sample_u), len(sample_v)
    D = scipy.spatial.distance_matrix(sample_u, sample_v, p = 2)
    K = np.exp(-D/epsilon)
    
    
    if u_weight == None:
        p_u = np.ones((m, 1))/m
    elif np.sum(u_weight) != 1:
        raise ValueError('sum of u_weight is not 1')
    else:
        p_u = np.array(u_weight)
    
    if v_weight == None:
        p_v = np.ones((n, 1))/n
    elif np.sum(v_weight) != 1:
        raise ValueError('sum of v_weight is not 1')
    else:
        p_v = np.array(v_weight)
    u_weight, v_weight = p_u, p_v
    
    P = np.diag(p_u.flatten()) @ K @ np.diag(p_v.flatten())
    p_norm = np.trace(P.T @ P)
    while True:
        p_u = u_weight/element_max(K @ p_v, 1e-300)
        p_v= v_weight/element_max(K.T @ p_u, 1e-300)
        P = np.diag(p_u.flatten()) @ K @ np.diag(p_v.flatten())
        if abs((np.trace(P.T @ P) - p_norm)/p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
    
    ans = np.trace(D.T @ P)
    print('sinkhorn answer: ', ans)
    return P, ans