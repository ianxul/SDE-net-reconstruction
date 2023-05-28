import numpy as np
import scipy.linalg as la
from scipy.optimize import linprog

# Function to generate constants with less memory usage
def alphas(i, j, l, k, U, u):
    return (U[i,l]*U[j,k] - (u[k]/u[l])*U[i,k]*U[j,l])
def betas(i, j, U, u):
    return sum([-U[i,s]*U[j,s]/(2*u[s]) for s in range(len(u))])

def get_A_sol(x_sol, n, U, u):
    A_sol = np.full((n,n), 0.0)
    for i in range(n):
        for j in range(n):
            A_sol[i,j] = betas(i,j, U, u) + sum(x_sol*np.array([alphas(i,j,l,k, U, u) for l in range(n) for k in range(l+1,n)]))
    return A_sol

# Linear programming to find solution. Assumption is that there are at least n(n-1)/2 zeroes.
def find_sol_lp(dt, E, verbose = True):
    n = E.shape[0]
    gamma = np.cov(dt)
    u, U = la.eig(gamma)

    ## Objective weights of the dims. These are chosen so that the weights of the edges have minimum sum.
    ut_n = (n*(n-1))//2
    m = 0
    # This loop counts the number of entries of E which are equal to zero
    for i in range(n):
        for j in range(n):
            # We don't optimize over non-existing edges
            if E[i,j] == 1.:
                continue
            m += 1
    
    # Vector used in the linear programming method. It determines the weight of the values of x when minimizing. 
    c = np.array([0.0]*ut_n + [1.0]*m)

    # Coefficients of the inequality constraint in the LP process. 
    A_ub = []
    b_ub = []
    m_count = 0
    for i in range(n):
        for j in range(n):
            # Existing edges don't give us restrictions
            if E[i,j] == 1.:
                continue
            A_ub.append([alphas(i,j,l,k, U, u) for l in range(n) for k in range(l+1, n)] + [-1.0*(i == m_count) for i in range(m)])
            b_ub.append(-betas(i,j, U, u))
            A_ub.append([(-alphas(i,j,l,k, U, u)) for l in range(n) for k in range(l+1, n)] + [-1.0*(i == m_count) for i in range(m)])
            b_ub.append(betas(i,j, U, u))
            m_count += 1
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    ## Equality constraints
    A_eq = None
    b_eq = None

    bounds = [(-10.0, 10.0) for _ in range(ut_n)] + [(-20.0, 20.0) for _ in range(m)]
    opt_res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    x_sol = opt_res.x[:ut_n]

    A_sol = get_A_sol(x_sol, n, U, u)
    if verbose: print(opt_res)
    return A_sol
 
# Find sol min-dis
def find_sol_lstsq(dt, E):
    n = E.shape[0]
    gamma = np.cov(dt)
    u, U = la.eig(gamma)

    phi_list = []
    for i in range(n):
        for j in range(n):
            if E[i,j] == 0:
                phi_list.append((i,j))
    
    zero_count = len(phi_list)


    ut_lst = []
    for i in range(n):
        for j in range(i+1,n):
            ut_lst.append((i,j))
    
    ut_m = len(ut_lst)

    Alpha_mat = np.full((zero_count, ut_m), 0.)
    for k in range(zero_count):
        i,j = phi_list[k]
        for r in range(ut_m):
            ll, kk = ut_lst[r]
            Alpha_mat[k,r] = alphas(i,j,ll,kk, U, u)

    beta_vec = np.full(zero_count, 0.)
    for k in range(zero_count):
        i,j = phi_list[k]
        beta_vec[k] = -betas(i,j, U, u)

    x_sol = la.lstsq(Alpha_mat, beta_vec)[0]

    return get_A_sol(x_sol, n, U, u)
    