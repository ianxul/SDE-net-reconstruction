import numpy as np

from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.estimators_opencl import OpenCLKraskovCMI

## Function that takes a source and target vector and an IDTxl TE estimator and calculates TE. Optionally more conditions can be passed along (not implemented yet). 
def estimate_CTE(source, target, TE_estimator, conditions = [], tau_source = 1, tau_target = 1):
    max_tau = max(tau_source, tau_target)
    return TE_estimator.estimate(target[max_tau:],source[(max_tau-tau_source):-max_tau],target[(max_tau-tau_target):-max_tau])


## This function will take data as a numpy array of shape (n,m) where n is the number of nodes and m the number of datapoints for each.
def generate_TE_mat(data, tau_source = 1, tau_target = 1, gpu = False):
    n, _ = data.shape

    TE_mat = np.full((n,n), 0.0)
    if not gpu:
        # If not using GPU use the JIDT estimator. Here we use CMI instead of TE for consistency with the OpenCL implementation.
        TE_estimator = JidtKraskovCMI()
    else:
        TE_estimator = OpenCLKraskovCMI()
    
    for i in range(n):
        for j in range(n):
            if i == j: 
                TE_mat[i,i] = 1.0
                continue
            ## The complicated bounds correspond to the restrictions given by the delays on each process
            TE_mat[i,j] = estimate_CTE(data[i,:], data[j,:], TE_estimator, tau_source = tau_source, tau_target = tau_target)

    return TE_mat
