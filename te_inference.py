import te_estimate
import numpy as np

from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.estimators_opencl import OpenCLKraskovCMI

## Functions related to finding directed edges

# Based on Rahimzamani & Kannan a first approach to testing significance is using gaussian variables.
def get_threshold(source_num = 1, points = 100000, reps = 100, gpu = False):
    if not gpu:
        # If not using GPU use the JIDT estimator. Here we use CMI instead of TE for consistency with the OpenCL implementation.
        TE_estimator = JidtKraskovCMI()
    else:
        TE_estimator = OpenCLKraskovCMI()
    thresh = 0
    for _ in range(reps):
        target = np.random.randn(points)
        source = np.random.randn(points)
        conds = np.random.randn((source_num-1, points))
        thresh += te_estimate.estimate_CTE(source, target, TE_estimator)