import numpy
import scipy
from wasserstein import wass_nd

'''
this script is used for testing different backends of the proposed wass_nd function.
backends including linear programming, 

'''

def wass_nd(u, v, u_weight = None, v_weight = None, method):
    '''

    '''
    ans = []
    for method_name in method:
        if method_name == 'linprog':
            ans.append(wass_nd(u, v, u_weight, v_weight))
        if method_name == 'network_flow':
            pass
        if method_name == 'sinkhorn':
            pass
        if method_name == 'Stoc_Opt_sinkhorn':
            pass
    
    return ans




