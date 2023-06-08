import numpy as np
import scipy.linalg as la
import cvxpy as cvx

# !These are not the same alphas and betas used in the other inference script.
## alphas define the restriction matrix for the rows corresponding to non diagonal entries in A bar.
def alphas(i, j, l, k, U, u):
    return u[j]*U[l,i]*U[k,j] + u[i]*U[l,j]*U[k,i]
## betas the rows corresponding to the diagonal entries
def betas(i, l, k, U):
    return U[l,i]*U[k,i]

# It seems like cvxpy does some additional reductions which make the running more efficient? Could also bypass and use SCS directly.
def find_sol_cvx(E : np.array, dt = None, gamma = None, verbose = False, eps_abs = 1e-2, max_iters = int(1e4)):
    n = E.shape[0]
    if not dt is None:
        gamma = np.cov(dt)
    elif gamma is None:
        raise Exception("Must provide either data or gamma")

    u, U = la.eig(gamma)

    Alpha_mat = np.full(((n*(n+1))//2,n**2), 0.0)
    beta_vec = np.full(((n*(n+1))//2), 0.0)
    zero_inds = np.full((n**2), 0)

    c1 = 0
    for i in range(n):
        for j in range(i,n):
            if i==j:
                beta_vec[c1] = -1/(2*u[i]).real
            c2 = 0
            for l in range(n):
                for k in range(n):
                    if E[l,k] == 0: zero_inds[c2] = 1
                    if (i!=j):
                        Alpha_mat[c1, c2] = alphas(i, j, l, k, U, u).real
                    else:
                        Alpha_mat[c1, c2] = betas(i, l, k, U).real
                    c2 += 1
            c1 += 1

    vx = cvx.Variable(n**2)
    objective = cvx.Minimize(cvx.norm(cvx.multiply(zero_inds,vx), 1))
    constraints = [Alpha_mat@vx == beta_vec]
    prob = cvx.Problem(objective, constraints)
    return prob.solve(verbose=verbose, solver = cvx.SCS, eps_abs = eps_abs, max_iters = max_iters)