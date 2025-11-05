import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import time

import scipy
from scipy.special import j0, jv
from scipy.signal.windows import blackman
from scipy.optimize import fsolve
from qutip import *


def ion_ham(Omega_arr, t_arr, eta, mu, omega, psi, N_cutoff, ham_type='full'):
    H_osc = omega*tensor(num(N_cutoff), qeye(n_ions+1))

    q = create(N_cutoff) + destroy(N_cutoff)


    if ham_type == 'full':
        exp_ietaq = (1j*eta*q).expm()
    elif ham_type == 'only_carrier':
        exp_ietaq = 1 + 1j*eta*q
    elif ham_type == 'LD':
        exp_ietaq = 1j*eta*q
    elif ham_type == 'LD2+carrier':
        exp_ietaq = 1 + 1j*eta*q - 0.5*eta**2*q**2

    Splus  = jmat(n_ions/2, '+')
    Sminus = jmat(n_ions/2, '-')

    return QobjEvo([H_osc, [-1j*tensor(exp_ietaq, Splus) + 1j*tensor(exp_ietaq.dag(), Sminus), Omega_arr*np.cos(mu*t_arr + psi)]],
            tlist=t_arr)


def blackman_mat(T, n_div):
    t_bp = np.linspace(0, T, n_div)
    b_vals = blackman(n_div)
    blackman_interp = CubicSpline(t_bp, b_vals)
    return t_bp, blackman_interp.c


def ghz_single_mode_infidelity(t_gate, n_ions, eta, omega, psi, n_t, N_cutoff, ham_type='full'):
    beta = 2.9999924
    gamma = 3.0486203
    delta = 2*np.pi*beta/t_gate
    mu = omega + delta
    Omega_max = np.pi*gamma/(eta*t_gate)*math.sqrt((mu + omega)/(2*omega))

    t_range = np.linspace(0, t_gate, n_t)

    Omega_arr = Omega_max*blackman(n_t)

    ham    = ion_ham(Omega_arr, t_range, eta, mu, omega, psi, N_cutoff, ham_type)
    psi_0  = tensor(fock(N_cutoff, 0), fock(n_ions+1, 0))
    result = sesolve(ham, psi_0, t_range)

    rhos = [s.ptrace([1]) for s in result.states]
    psi_id = 1/math.sqrt(2)*(fock(n_ions+1, 0) + (-1)**(n_ions/2)*1j*fock(n_ions+1, n_ions))

    F = psi_id.dag() * rhos[-1] * psi_id
    return 1-F.real


if __name__ == '__main__':
    psi = 0
    dt = 0.1

    t_gate = 150
    n_ions = 2

    nu = 0.7
    omega= 2*np.pi*nu
    N_cutoff = 14

    nu_rec = 9394*1e-6
    eta = np.sqrt(2*np.pi*nu_rec/(n_ions*omega))

    n_t = int(t_gate/dt)
    start_time = time.time()
    inf = ghz_single_mode_infidelity(t_gate, n_ions, eta, omega, psi, n_t, N_cutoff, ham_type='full')
    end_time = time.time()

    print(f"time= {end_time-start_time}")
    print(f' t_gate = {t_gate}, inf={inf}')
