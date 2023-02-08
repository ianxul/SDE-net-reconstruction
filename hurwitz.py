import numpy as np

from jitcsde import jitcsde, y

# Function to generate a random Hurtzian matrix (not yet implemented)
def random_hurwitz() -> np.matrix:
    return np.matrix([
            [-1.0,0.0, -1.0],
            [0.5, -2.0, 0.0],
            [0.0, -1.0, -1.0]
        ])

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