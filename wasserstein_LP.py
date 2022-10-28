import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from test_utils import *

# todo: try with sparse matrix?

@time_it
def wass_nd(sample_u, sample_v, u_weight = None, v_weight = None):
    '''
    n dimensional wasserstein function
    '''
    m, n = len(sample_u), len(sample_v)
    
    row, col, value = [], [], []
    
    for i in range(m):
        for j in range(n):
            row.append(i)
            col.append(n*i + j)
            value.append(1)
    for i in range(m):
        for j in range(n):
            row.append(m + j)
            col.append(n*i + j)
            value.append(1)

    A = coo_matrix((value, (row, col)), shape=(n+m, m*n)) 
    D = scipy.spatial.distance_matrix(sample_u, sample_v, p = 2)
    c = D.reshape((n*m))
    
    if u_weight == None:
        p_u = [1/m for i in range(m)]
    elif np.sum(u_weight) != 1:
        raise ValueError('sum of u_weight is not 1')
    else:
        p_u = u_weight
    
    if v_weight == None:
        p_v = [1/n for i in range(n)]
    elif np.sum(v_weight) != 1:
        raise ValueError('sum of v_weight is not 1')
    else:
        p_v = v_weight
    
    b = np.concatenate((p_u, p_v), axis = 0)
    opt_res = linprog( -b , A_ub = A.T, b_ub = c, bounds=(None, None))
    ans = b @ opt_res.x
    print('LP answer:', ans)
    return ans