import numpy as np
import hurwitz
import scipy.linalg as la
from scipy.optimize import linprog
from itertools import combinations

def get_gamma(dt):
    return sum([np.dot(np.transpose([dt[i,]]), np.array([dt[i,]])) for i in range(len(dt[:,0]))])/len(dt[:,0])

def calc_consts(n, U, u):
    alphas = np.full((n,n,n,n), 0.0)
    for i in range(n):
        for j in range(n):
            for l in range(n):
                for k in range(n):
                    alphas[i,j,l,k] = (U[i,l]*U[j,k] - (u[k]/u[l])*U[i,k]*U[j,l])

    betas = np.full((n,n), 0.0)
    for i in range(n):
        for j in range(n):
            betas[i,j] = sum([-U[i,s]*U[j,s]/(2*u[s]) for s in range(n)])

    return alphas, betas

def get_A_sol(x_sol, n, alphas, betas, E):
    A_sol = np.full((n,n), 0.0)
    for i in range(n):
        for j in range(n):
            # if E[i,j] == 0.:
            #     continue
            A_sol[i,j] = betas[i,j] + sum(x_sol*np.array([alphas[i,j,l,k] for l in range(n) for k in range(l+1,n)]))
    return A_sol

# Linear programming to find solution. Assumption is that there are at least n(n-1)/2 zeroes.
def find_sol_lp(dt, E, verbose = True):
    n = E.shape[0]
    gamma = get_gamma(dt)
    u, U = la.eig(gamma)

    alphas, betas = calc_consts(n, U, u)

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
    
    c = np.array([0.0]*ut_n + [1.0]*m)

    A_ub = []
    b_ub = []
    m_count = 0
    for i in range(n):
        for j in range(n):
            # Existing edges don't give us restrictions
            if E[i,j] == 1.:
                continue
            A_ub.append([alphas[i,j,l,k] for l in range(n) for k in range(l+1, n)] + [-1.0*(i == m_count) for i in range(m)])
            b_ub.append(-betas[i,j])
            A_ub.append([(-alphas[i,j,l,k]) for l in range(n) for k in range(l+1, n)] + [-1.0*(i == m_count) for i in range(m)])
            b_ub.append(betas[i,j])
            m_count += 1
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    ## Equality constraints
    A_eq = None
    b_eq = None

    bounds = [(-10.0, 10.0) for _ in range(ut_n)] + [(-20.0, 20.0) for _ in range(m)]
    opt_res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    x_sol = opt_res.x[:ut_n]

    A_sol = get_A_sol(x_sol, n, alphas, betas, E)
    if verbose: print(opt_res)
    return A_sol
 
# Find sol min-dis
def find_sol_lstsq(dt, E):
    n = E.shape[0]
    gamma = get_gamma(dt)
    u, U = la.eig(gamma)

    alphas, betas = calc_consts(n, U, u)

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
            Alpha_mat[k,r] = alphas[i,j,ll,kk]

    beta_vec = np.full(zero_count, 0.)
    for k in range(zero_count):
        i,j = phi_list[k]
        beta_vec[k] = -betas[i,j]
    
    # sol_sys = np.matmul(Alpha_mat.T, Alpha_mat)
    # sol_vec = np.matmul(Alpha_mat.T, beta_vec)

    # x_sol = la.solve(sol_sys, sol_vec)

    x_sol = la.lstsq(Alpha_mat, beta_vec)[0]

    return get_A_sol(x_sol, n, alphas, betas, E)

# Solving the problem under the assumption that the number of zeroes is exactly n(n-1)/2.
def find_sol_la(dt, E):
    n = E.shape[0]
    gamma = get_gamma(dt)
    u, U = la.eig(gamma)

    alphas, betas = calc_consts(n, U, u)

    ## Equality constraints
    A_eq = []
    b_eq = []
    for i in range(n):
        for j in range(n):
            # Existing edges don't give us restrictions
            if E[i,j] == 1.:
                continue
            A_eq.append([alphas[i,j,l,k] for l in range(n) for k in range(l+1, n)])
            b_eq.append(-betas[i,j])
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    o_num = len(b_eq)

    combs = combinations(range(o_num), (n*(n-1))//2)
    xs = []
    for comb in combs:
        comb = list(comb)
        x = la.solve(A_eq[comb, :], b_eq[comb])
        # print("x is {}".format(x))
        xs.append(x)

    # Average of solutions
    # x_sol = sum(xs)/len(xs)
    # print("Average x is {}".format(x_sol))

    A_sols = [get_A_sol(x_sol, n, alphas, betas, E) for x_sol in xs]
    A_sol = sum(A_sols)/len(A_sols)
    
    # print(opt_res)
    return A_sol

def run_test(A, dt = None, time = 1000, step = 0.01):
    if not np.any(dt):
        dt = hurwitz.run_process(A, time, step).T
    E = np.abs(np.sign(A))
    A_sol = find_sol_lp(dt, E)
    print("Test matrix was:")
    print(A)
    print("Sol matrix was:")
    print(A_sol)
    

    