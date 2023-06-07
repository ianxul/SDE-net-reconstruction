import te_estimate
import numpy as np

from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.estimators_opencl import OpenCLKraskovCMI

import heapq
import pickle
import os

## Functions related to finding directed edges

# Based on Rahimzamani & Kannan a first approach to testing significance is using gaussian variables.
def get_threshold(alpha, TE_estimator, source_num = 1, points = 100000, reps = 100):
    # Create a lits of TE values from surrogate data (for now gaussian)
    vals = []
    for _ in range(reps):
        target = np.random.randn(points)
        source = np.random.randn(points)
        conds = np.random.randn(points, source_num-1)
        vals.append(te_estimate.estimate_CTE(source, target, TE_estimator, conditions = conds))

    # Sort in decreasing order
    vals.sort(reverse = True)

    # Return threshold value
    return vals[int(alpha*reps)]

def perform_inference(dt, alpha = 0.05, test_reps = 100, gpu = False, limit = None, report_edges = True):
    n, m = dt.shape
    if not limit:
        limit = (n*(n-1))//2

    if not gpu:
        # If not using GPU use the JIDT estimator. Here we use CMI instead of TE for consistency with the OpenCL implementation.
        TE_estimator = JidtKraskovCMI()
    else:
        TE_estimator = OpenCLKraskovCMI()

    # To be more efficient we store the threshold values found
    threshold_file_name = "pickles/thresh_{}alpha_{}points_{}reps.pkl".format(int(alpha*100), m, test_reps)
    if os.path.isfile(threshold_file_name):
        thresholds_file = open(threshold_file_name, 'rb')
        thresholds = pickle.load(thresholds_file)
        thresholds_file.close()
    else:   
        thresholds = [0, get_threshold(alpha, TE_estimator, points = m, reps = test_reps)]

    # Adjacency matrix starts with all recurrent connections
    adj_mat = np.identity(n,int)
    count = 0

    ## More conditioning on variables can only reduce entropy, so the algorithm can benefit from using a heap
    ## Fill in heaps. Elements of heap_lst will be tuples of the form (node num, node heap)
    heap_lst = []
    for i in range(n):
        node_heap = []
        for j in range(n):
            if i == j:
                continue
            heapq.heappush(node_heap, (-te_estimate.estimate_CTE(dt[j,:], dt[i,:], TE_estimator), j, 1))
        heap_lst.append((i, node_heap))
    
    sources_lst = [[] for _ in range(n)]

    while (count < limit) and len(node_heap):
        while not len(heap_lst[-1][1]):
            heap_lst.pop()
        
        for (i, node_heap) in heap_lst:
            if not len(node_heap): continue
            te_val, j, source_num = heapq.heappop(node_heap)
            te_val *= -1.0

            while source_num > len(thresholds)-1:
                thresholds.append(get_threshold(alpha, TE_estimator, source_num = len(thresholds), points = m, reps = test_reps))
            
            if te_val > thresholds[source_num]:
                if report_edges: print("Adding ({}->{}) edge with TE: {}".format(j,i,te_val))
                sources_lst[i].append(j)
                adj_mat[i, j] = 1 # important to notice the order is the oposite as in adj matrix
                count += 1
                if limit <= count:
                    break
            else: 
                node_heap.clear()
                continue

            while len(node_heap) and node_heap[0][2] < source_num+1:
                _, k, k_sn = heapq.heappop(node_heap)
                heapq.heappush(node_heap, (-te_estimate.estimate_CTE(dt[k,:], dt[i,:], TE_estimator, conditions = dt[sources_lst[i],:].T), k, k_sn+1))
    
    # Save calculated values of thresholds
    thresholds_file = open(threshold_file_name, 'wb')
    pickle.dump(thresholds, thresholds_file)
    thresholds_file.close()

    return adj_mat

def perform_inference_max(dt, gpu = False):
    n, m = dt.shape
    limit = (n*(n-1))//2

    if not gpu:
        # If not using GPU use the JIDT estimator. Here we use CMI instead of TE for consistency with the OpenCL implementation.
        TE_estimator = JidtKraskovCMI()
    else:
        TE_estimator = OpenCLKraskovCMI()

    # Adjacency matrix starts with all recurrent connections
    adj_mat = np.identity(n,int)
    te_vals = np.zeros((n,n))
    count = 0

    ## Fill heap with TE values.
    node_heap = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            heapq.heappush(node_heap, (-te_estimate.estimate_CTE(dt[j,:], dt[i,:], TE_estimator), i, j))
    
    while count < limit:
        te_val, i, j = node_heap.pop()
        adj_mat[i,j] = 1
        te_vals[i,j] = te_val
        count += 1
    
    return adj_mat, te_vals
            

def is_success(A, E):
    E_real = np.abs(np.sign(A))
    return np.array_equal(E_real, np.multiply(E_real, E))