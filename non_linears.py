import numpy as np

from jitcsde import jitcsde, y

# Lorenz
def gen_lorenz(pars = [10.0,28.0,2.66], time_length = 100, step = 0.01, noise = 1.):
    sigma, rho, beta = pars
    diff_sys = [sigma*(y(1)-y(0)), y(0)*(rho-y(2))-y(1), y(0)*y(1) - beta*y(2)]

    # Stochastic part of equation
    noise_term = [noise for _ in range(3)]

    SDE = jitcsde(diff_sys, noise_term)

    initial_state = np.array([1.0, 0.0, 0.0])
    SDE.set_initial_value(initial_state, 0.0)

    data = []
    for time in np.arange(0.0, time_length, step):
        data.append(SDE.integrate(time))

    return np.array(data)