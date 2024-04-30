"""
Solve the decay equation with RK45. 
This method converges, but requires extremely small dt if A is stiff.

The equation is 
dN/dt = A*N(t) with N(0) = N0
"""

import numpy as np
from scipy.integrate import RK45

def RK45solve(A, N0, t, atol = 1e-8, rtol = 1e-5):
    RHS = lambda t, N: A.dot(N)
    solver = RK45(RHS, t0 = 0.0, y0 = N0, t_bound = t, atol = atol, rtol = rtol)
    while solver.t < t:
        solver.step()
    # print("RK solver finished at time", solver.t)
    return solver.y

def RK45solve_full_trajectory(A, N0, t, atol = 1e-8, rtol = 1e-5):
    RHS = lambda t, N: A.dot(N)
    solver = RK45(RHS, t0 = 0.0, y0 = N0, t_bound = t, atol = atol, rtol = rtol)
    tsRK = []
    NsRK = []
    while solver.t < t:
        solver.step()
        tsRK.append(solver.t)
        NsRK.append(solver.y)
    tsRK = np.array(tsRK)
    NsRK = np.array(NsRK)
    return tsRK, NsRK