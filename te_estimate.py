import numpy as np

from idtxl.estimators_jidt import JidtKraskovTE
from idtxl.estimators_opencl import OpenCLKraskov

def get_TE_estimator(settings, gpu = False):
    if not gpu:
        return JidtKraskovTE(settings = settings)

## This function will take data as a numpy array of shape (n,m) where n is the number of nodes and m the number of datapoints for each.
def generalte_TE_mat(data, history_source = 1, history_target = 1, delay = 1, tau_source = 1, tau_target = 1, gpu = False):
    n, m = data.shape

    TE_mat = np.full((n,n), 0.0)

    settings = dict({
        'history_source' : history_source,
        'history_target' : history_target,
        'tau_source' : tau_source,
        'tau_target' : tau_target,
        'source_target_delay' : delay
    })
    TE_estimator = get_TE_estimator(settings, gpu)

    for i in range(n):
        for j in range(n):
            if i == j: 
                TE_mat[i,i] = 1.0
                continue
            TE_mat[i,j] = TE_estimator.estimate(data[i,:],data[j,:])

    return TE_mat
