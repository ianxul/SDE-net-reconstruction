import numpy as np
import networkx as nx

from jitcsde import jitcsde, y

# This is a workaround to run Julia from Python interactively. However, this takes in general a couple minutes.
from julia.api import Julia
print("Starting up Julia...")
jl = Julia(compiled_modules=False)
from diffeqpy import de

# Function to generate a random Hurwitz matrix. In this function the number of edges is restricted to be less than the upper triangle number. There is no restriction on symmetric connections. We used this method first but the results were too good, it seemed like these matrices were too easy to reconstruct. 
def random_hurwitz(n:int, ep:float = 0.2, allow_underdet:bool = False) -> np.matrix:
    # Random directed graph. p = 2/n seems reasonable to keep graph sparce.
    G = nx.fast_gnp_random_graph(n, p = ep/n, directed=True)
    # While the graph has too many edges keep repeating.
    while not len(G.edges) < (n*(n-1))//2 and not allow_underdet:
        G = nx.fast_gnp_random_graph(n, p = ep/n, directed=True)

    A_mat = np.full((n,n), 0.)
    # Assign random weight to all edges according to normal
    for e in G.edges:
        A_mat[e[0],e[1]] = np.random.randn()
    
    # Assign diagonal entries accorgind to Gershgorin disk theorem to ensure stability.
    for ii in range(n):
        A_mat[ii,ii] = -np.sum(np.abs(A_mat[ii,:])) - 0.1
        # If node has no incomming edges assign value of -1.
        if A_mat[ii,ii] == -0.1:
            A_mat[ii,ii] -= 0.9

    return A_mat

# Function to generate a random Hurwitz matrix. In this function the number of edges in the Erdos-Renyi is restricted to be equal to parameter edm. Due to the dominant dynamics this example was much harder for our algorithm, particularly for the "No Info" method.
def random_hurwitz_dominant(n:int, edm:int, eps:float, max_reig_rad:float = 0.1) -> np.matrix:
    while 1:
        # Random directed graph
        G = nx.gnm_random_graph(n, edm, directed=True)
        B = np.full((n,n), 0.)
        # Assign random weight to all edges according to normal
        for e in G.edges:
            B[e[0],e[1]] = np.random.randn()

        eigs = np.linalg.eigvals(B)
        # Max real part of eigenvalues
        max_reig = np.max(np.real(eigs))
        # If the max real part is larger than the max_reig_rad parameter, break the loop. This was chosen to ensure stability.
        if abs(max_reig) > max_reig_rad:
            break

    # Create Hurwitz matrix
    B = -np.eye(n) + (1-eps)*B/max_reig
    return B

# Function to generate time-series data from a given Hurwitz matrix. The function uses the jitcsde package to solve the differential equation. Returns time-series data as a numpy array. This is an alternative to the run_process_jl function, but is much slower and allows less control over the integration process.
def run_process(A_mat:np.matrix, time_length:float, step:float = 0.1, noise:float = 1., saturating:bool = False, verbose:bool = True, diff_sys = None) -> np.array:
    N = len(A_mat)
    # Define deterministic part of differential equation
    if diff_sys is None:
        diff_sys = [sum([A_mat[j,i]*y(j) for j in range(N)]) for i in range(N)]
    if verbose: print(diff_sys)

    # Stochastic part of equation
    noise_term = [noise for _ in range(N)]

    SDE = jitcsde(diff_sys, noise_term, verbose = verbose)

    initial_state = np.array([0.0  for _ in range(N)])
    SDE.set_initial_value(initial_state, 0.0)

    data = []
    for time in np.arange(0.0, time_length, step):
        data.append(SDE.integrate(time))

    return np.array(data)


# Function to generate time-series data from a given Hurwitz matrix. Uses julia to solve the differential equation. Returns time-series data as a numpy array. This is an alternative to the run_process function, but is much faster and allows more control over the integration process.
def run_process_jl(A_mat:np.matrix, time_length:float, step:float = 0.1, noise:float = 1., saturating:bool = False, process_step = False) -> np.array:
    n = A_mat.shape[0]
    
    # Differential equation has two cases: saturating and non-saturating.
    if not saturating:
        def f(du, u, p, t):
            A, n, _ = p
            for i in range(n):
                du[i] = sum([A[i,j]*u[j] for j in range(n)])
    else: 
        def f(du, u, p, t):
            A, n, _ = p
            for i in range(n):
                # du[i] = np.tanh(sum([A[i,j]*u[j] for j in range(n)])) # First version of saturation
                du[i] = A[i,i]*u[i] + np.sum([np.tanh(A[i,j]*u[j]) for j in range(n) if i != j])

    # Stochastic part of equation
    def g(du, u, p, t):
        _, _, noise = p
        for i in range(n):
            du[i] = noise
    
    # Couldn't get numba just-in-time to work so just used the original functions.
    # numba_f = numba.jit(f)
    # numba_g = numba.jit(g)
    numba_f = f
    numba_g = g

    # Initial conditions
    u0 = np.zeros(n)
    # Time span
    tspan = (0.0, time_length)
    # Parameters
    p = [A_mat, n, noise]
    
    # Solve the differential equation
    prob = de.SDEProblem(numba_f, numba_g, u0, tspan, p)
    if not process_step:
        sol = de.solve(prob, de.LambaEM(), saveat = step)
    else: 
        sol = de.solve(prob, de.EM(), dt = process_step, saveat = step)

    return np.transpose(sol.u)


# Not very efficient but useful for small matrices to compare simulated gamma with the ground truth covariance matrix.
def analytic_gamma(A):
    n = A.shape[0]
    return np.reshape(-np.matmul(np.linalg.inv(np.kron(A, np.identity(n)) + np.kron(np.identity(n), A)), np.matrix.flatten(np.identity(n))), (n,n))

# Function to calculate the distance between two matrices. This does not give reliable measures of performance for the non-linear case since results are usually scaled with respect to the original. The ignore_diag parameter allows to ignore the diagonal entries of the matrices.
def calc_mat_dist(mat1, mat2, ignore_diag = True):
    n = mat1.shape[0]
    if ignore_diag:
        mat1 = mat1*(np.ones((n,n))-np.identity(n))
        mat2 = mat2*(np.ones((n,n))-np.identity(n))
    return np.sum(np.abs(mat1-mat2))/(n**2 - ignore_diag*n)

# Function to calculate alignment (or cos) between two matrices. The ignore_diag parameter allows to ignore the diagonal entries of the matrices.
def calc_mat_cos(mat1, mat2, ignore_diag = True):
    n = mat1.shape[0]
    if ignore_diag:
        mat1 = mat1*(np.ones((n,n))-np.identity(n))
        mat2 = mat2*(np.ones((n,n))-np.identity(n))
    if np.sum(mat1*mat2)==0:
        return 1.
    return np.sum(mat1*mat2)/(np.sqrt((mat1*mat1).sum())*np.sqrt((mat2*mat2).sum()))