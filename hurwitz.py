import numpy as np
import networkx as nx

from jitcsde import jitcsde, y

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Function to generate a random Hurtzian matrix. In this function the number of edges is restricted to be less than the upper triangle number. There is no restriction on symmetric connections.
def random_hurwitz(n) -> np.matrix:
    # Random directed graph. p = 2/n seems reasonable to keep graph sparce.
    # While the graph has too many edges keep repeating.
    while True:
        G = nx.fast_gnp_random_graph(n, p = 2./n, directed=True)
        if len(G.edges) < (n*(n-1))//2: break

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

# Same as above but with added restriction to non symmmetric connections.
def random_hurwitz_nonsym(n) -> np.matrix:
    
    while True:
        G = nx.fast_gnp_random_graph(n, p = 2./n)
        if len(G.edges) < (n*(n-1)): break
    A_mat = np.full((n,n), 0.)

    # Assign random weight to one of the two directions. If previously assigned skip.
    for e in G.edges:
        if A_mat[e[0],e[1]] or A_mat[e[1],e[0]]: continue
        bernie = np.random.randint(0,2)
        A_mat[e[bernie],e[1-bernie]] = np.random.randn()
    # Assign diagonal entries.
    for ii in range(n):
        A_mat[ii,ii] = -np.sum(np.abs(A_mat[ii,:])) - 0.1
        if A_mat[ii,ii] == -0.1:
            A_mat[ii,ii] -= 0.9
    return A_mat

def run_process(A_mat:np.matrix, time_length:float, step:float = 0.1, noise = 1., verbose = True, diff_sys = None) -> np.array:
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

# Not very efficient but useful for small matrices. 
def analytic_gamma(A):
    n = A.shape[0]
    return np.reshape(-np.matmul(np.linalg.inv(np.kron(A, np.identity(n)) + np.kron(np.identity(n), A)), np.matrix.flatten(np.identity(n))), (n,n))

def plot_A_img(A):
    A_bound = np.max(np.abs(A))
    plt = px.imshow(A, -A_bound, A_bound)
    plt.show()

def make_triple_imshow(A1, A2, A3, titles = ["A", "Linear Programming", "Least Squares"]):
    bound_val = np.max([np.max(np.abs(A)) for A in [A1,A2,A3]])
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
    fig.add_trace(
        go.Heatmap(z=A1, zmax = bound_val, zmin = -bound_val, zmid = 0, showscale=True, colorscale="RdBu", reversescale=True),
        row = 1, col = 1
    )
    fig.add_trace(
        go.Heatmap(z=A2, zmax = bound_val, zmin = -bound_val, zmid = 0, showscale=False, colorscale="RdBu", reversescale=True),
        row = 1, col = 2
    )
    fig.add_trace(
        go.Heatmap(z=A3, zmax = bound_val, zmin = -bound_val, zmid = 0, showscale=False, colorscale="RdBu", reversescale=True),
        row = 1, col = 3
    )
    fig.update_yaxes(autorange = 'reversed')
    
    fig.show()

def make_full_imshow(As, titles):
    bound_val = np.max([np.max(np.abs(A)) for A in As])
    fig = make_subplots(rows=1+np.ceil((len(As)-1)/3), cols=3, subplot_titles=titles)
    fig.add_trace(
        go.Heatmap(z=As[0], zmax = bound_val, zmin = -bound_val, zmid = 0, showscale=True, colorscale="RdBu", reversescale=True),
        row = 1, col = 1
    )
    for i in range(len(As)-1):
        fig.add_trace(
            go.Heatmap(z=As[i+1], zmax = bound_val, zmin = -bound_val, zmid = 0, showscale=False, colorscale="RdBu", reversescale=True),
            row = 2+(i//3), col = 1+(i%3)
        )
    
    fig.update_yaxes(autorange = 'reversed')
    
    fig.show()

def calc_mat_dist(mat1, mat2):
    n = mat1.shape[0]
    return np.sum(np.abs(mat1-mat2))/n**2