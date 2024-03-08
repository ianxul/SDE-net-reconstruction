import numpy as np

from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.estimators_opencl import OpenCLKraskovCMI

## Function that takes a source and target vector and an IDTxl TE estimator and calculates TE. Optionally more conditions can be passed along. 
def estimate_CTE(source, target, TE_estimator, conditions = np.array([]), tau_source = 1, tau_target = 1, tau_conditions = 1):
    max_tau = max(tau_source, tau_target, tau_conditions)
    if not np.any(conditions):
        conditions_array = target[(max_tau-tau_target):-tau_target]
    else:
        conditions_array = np.column_stack([
                target[(max_tau-tau_target):-tau_target],
                conditions[(max_tau-tau_conditions):-tau_conditions,:]
            ])
    TE_est = TE_estimator.estimate(
        target[max_tau:],
        source[(max_tau-tau_source):-tau_source],
        conditions_array
        )
    if isinstance(TE_est,float):
        return TE_est
    else:
        return TE_est[0]


## This function will take data as a numpy array of shape (n,m) where n is the number of nodes and m the number of datapoints for each.
## It will return a matrix of shape (n,n) where the element (i,j) is the estimated TE from j to i.
def generate_TE_mat(data, E = None, tau_source = 1, tau_target = 1, gpu = False):
    n, _ = data.shape

    TE_mat = np.full((n,n), 0.0)
    if not gpu:
        # If not using GPU use the JIDT estimator. Here we use CMI instead of TE for consistency with the OpenCL implementation.
        TE_estimator = JidtKraskovCMI()
    else:
        TE_estimator = OpenCLKraskovCMI()
    
    for i in range(n):
        source_lst = []
        if not E is None:
            for j in range(n):
                if i==j:continue
                if E[i,j] == 1:
                    source_lst.append(j)
        for j in range(n):
            if i == j: 
                TE_mat[i,i] = 1.0
                continue
            ## The complicated bounds correspond to the restrictions given by the delays on each process
            TE_mat[j,i] = estimate_CTE(data[j,:], data[i,:], TE_estimator, conditions = data[[k for k in source_lst if k!=j],:].T, tau_source = tau_source, tau_target = tau_target)

    return TE_mat
