import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integ
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import Bounds

#Let's first find the configuration of the rods for which F is minimum (equilibrium condition)
#We set gamma = alpha = 1
def F1 (Ns, V):
    alpha = 1
    gamma = 1
    Nx, Ny, Nz = Ns[0], Ns[1], Ns[2]
    F = Nx*np.log(alpha*Nx/V) + Ny*np.log(alpha*Ny/V) + Nz*np.log(alpha*Nz/V) + gamma*(Nx*Ny + Ny*Nz + Nz*Nx)/V
    return F

def F1_1D (Nx, N, V):
    alpha = 1
    gamma = 1
    Ns = np.array([Nx, (N - Nx)/2, (N - Nx)/2])
    F = F1(Ns, V)
    return F

def F1_1D_V (V, Nx, N):
    alpha = 1
    gamma = 1
    Ns = np.array([Nx, (N - Nx)/2, (N - Nx)/2])
    F = F1(Ns, V)
    return F

#Returns the n_x which minimizes F
def Get_n_min(N, V):
    Nx_0 = N*0.5
    b = Bounds([0], [N])
    res = minimize(F1_1D, Nx_0, args=(N, V), bounds=b, method = 'trust-constr')
    return float(res.x/N)

def Get_n_min_s(N_s, V):
    n_x_s = []
    for N in N_s:
        n_x_s.append(Get_n_min(N, V))
    return n_x_s

def Get_F_eq (N, V):
    N_x = N*Get_n_min(N, V)
    return F1_1D_V(V, Nx, N)

def Get_P (N, V):
    n_x = Get_n_min(N, V)
    P = derivative(F1_1D_V, V, n_x*N, N)
    return -P

def Get_P_s (Ns, V):
    Ps = []
    for N in Ns:
        Ps.append(Get_P(N, V))
    return Ps

def Get_P_s_V (Vs, N):
    Ps = []
    for V in Vs:
        Ps.append(Get_P(N, V))
    return Ps

def derivative(f, a, par1, par2, method='central', h=0.001):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    par1, par2 : parameter
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula         
    '''
    if method == 'central':
        return (f(a + h, par1, par2) - f(a - h, par1, par2))/(2*h)
    elif method == 'forward':
        return (f(a + h, par1, par2) - f(a, par1, par2))/h
    elif method == 'backward':
        return (f(a, par1, par2) - f(a - h, par1, par2))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
        
#Gibbs' Free Energy
def G (x_s_cut, y_s_cut):
    x_s_1 = np.delete(x_s_cut, 0)
    x_s_2 = np.delete(x_s_cut, -1)
    delta_xs = x_s_1 - x_s_2
    y_s_1 = np.delete(y_s_cut, 0)
    t_s = delta_xs * y_s_1
    return np.sum(t_s)

def G_s (P_s, V_s):
    G_s = [0] #the area under the first point: =0!
    for i in range(1, len(P_s), 1):
        P = P_s[i]
        G_s.append(G(P_s[P_s<=P], V_s[P_s<=P])) #I evaluate the integral until each point 
    return G_s