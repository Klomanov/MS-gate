import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import CubicSpline, PPoly
from timslib.ion_crystals import IonChain
from timslib.ion_crystals.normal_modes import nu_ax_from_nu_rad_min
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timslib.ms_pulse_shaping.ms_evo_funcs.ppoly_alpha_and_chi import *

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def smoothed_step_mat_from_t(t_gate, sm_deg):
    
    def smoothed_step_mat(t_mid, tau):
        t_bp = np.array([0, tau, t_mid + tau, t_mid + 2*tau])
        omega_mat = np.array([[-2/tau**3, 3/tau**2, 0, 0], [0, 0, 0, 1], [2/tau**3, -3/tau**2, 0, 1]]).T

        return t_bp, omega_mat
    
    tau = sm_deg * t_gate
    t_mid = t_gate - 2 * tau
    t_bp, omega_mat = smoothed_step_mat(t_mid, tau)
    delta = 1 / (t_gate * (1 - sm_deg))

    return delta, t_bp, omega_mat

def multitone_chi_alpha(t1, t2, t_bp, mu_Omega_psi_list, omegas, omega_mat):
    
    def multitone_chi_alpha_at_bp(t_bp, Omega_mat, mu, psi, omegas):
        return ppoly_alpha_eqf_at_bp(t_bp, Omega_mat, mu, psi, omegas)

    val = 0
    for mu, Omega, psi in mu_Omega_psi_list:
        mode_alpha_at_bp = multitone_chi_alpha_at_bp(t_bp, Omega * omega_mat, mu, psi, omegas)
        val += ppoly_alpha_eqf(t2, t1, t_bp, Omega * omega_mat, mu, psi, omegas, mode_alpha_at_bp)

    return val

def multitone_chi_mode(t1, t2, t_bp, mu_Omega_psi_list, omegas, omega_mat):

    def re_integrand(t):
        val = 0
        for mu, Omega, psi in mu_Omega_psi_list:
            Omega_func = PPoly.construct_fast(Omega * omega_mat, t_bp)
            val += Omega_func(t) * np.cos(mu * t + psi) * np.exp(-1j * omegas * t)
        alpha = multitone_chi_alpha(t1, t, t_bp, mu_Omega_psi_list, omegas, omega_mat)

        return np.real(val * alpha)
        
    def im_integrand(t):
        val = 0
        for mu, Omega, psi in mu_Omega_psi_list:
            Omega_func = PPoly.construct_fast(Omega * omega_mat, t_bp)
            val += -Omega_func(t) * np.cos(mu * t + psi) * np.exp(-1j * omegas * t)
        alpha = multitone_chi_alpha(t1, t, t_bp, mu_Omega_psi_list, omegas, omega_mat)

        return np.imag(val * alpha)
        
    chi_m = np.real(2 * (scipy.integrate.quad_vec(re_integrand, t1, t2)[0] + 1j * scipy.integrate.quad_vec(im_integrand, t1, t2)[0]))
    return chi_m

n_ions = 8
nu_rad = 1.6e6
nu_ax = 0.15e6 #0.41
 
t_gate = 100e-6
gamma = 1.11134871

t1 = 0
t2 = t_gate

sm_deg = 0.1

chain = IonChain(nu_rad=nu_rad, nu_ax=nu_ax, n_ions=n_ions, ion_type='Ca40')

eta = chain.eta_ax(0)[0,0]
eta_mat = chain.eta_ax(0)

delta, t_bp, omega_mat = smoothed_step_mat_from_t(t_gate, sm_deg)

Omega0 = gamma * np.pi / (eta * t_gate)
Omega1 = 0.29 * Omega0
Omega2 = 0.75 * Omega1
Omega3 = 0.78 * Omega2
Omega4 = 0.79 * Omega3

mu0 = 2 * np.pi * (nu_ax - delta)
mu1 = chain.omegas_ax[1] + 2 * np.pi * delta
mu2 = chain.omegas_ax[2] + 2 * np.pi * delta
mu3 = chain.omegas_ax[3] + 2 * np.pi * delta
mu4 = chain.omegas_ax[4] + 2 * np.pi * delta

# mu_Omega_psi_list = [[mu0, Omega0, 0], [mu1, Omega1, 0]]#, [mu2, Omega2, 0], [mu3, Omega3, 0]]#, [mu4, Omega4, 0]]

# chi_mode = multitone_chi_mode(t1, t2, t_bp, mu_Omega_psi_list, chain.omegas_ax, omega_mat)
# chi_matrix = eta_mat @ np.diag(chi_mode) @ eta_mat.T
# chi_no_diag = chi_matrix * (1 + np.diag(np.ones(n_ions)*np.nan))
# chi_average = np.sum(np.tril(chi_no_diag, k=-1))/(n_ions*(n_ions-1)/2)
# infidelity = np.sum(np.tril((chi_no_diag/chi_average - 1)*np.pi/4, k=-1)**2)

# print('Infidelity = ', infidelity)
# print('Alpha = ', multitone_chi_alpha(t1, t2, t_bp, mu_Omega_psi_list, chain.omegas_ax, omega_mat))
# print('Eta = ', eta_mat)

# plt.gca().set_aspect('equal')
# plt.pcolormesh(np.abs(chi_no_diag/(np.pi/4)), norm=matplotlib.colors.CenteredNorm(1), cmap=plt.get_cmap('seismic', 100))
# plt.colorbar()
# plt.clim(0.85,1.15)
# plt.title('$4\\chi_{ij}/\\pi$')
# plt.xlabel('$i$')
# plt.ylabel('$j$')
# plt.show()

resol = 100
inf_arr = np.zeros((resol))
omega_arr = np.zeros((resol))
for i in range (0, resol):
    print(i)
    Omega3 = (i/100)*Omega2
    mu_Omega_psi_list = [[mu0, Omega0, 0], [mu1, Omega1, 0], [mu2, Omega2, 0], [mu3, Omega3, 0]]#, [mu4, Omega4, 0]]

    chi_mode = multitone_chi_mode(t1, t2, t_bp, mu_Omega_psi_list, chain.omegas_ax, omega_mat)
    chi_matrix = eta_mat @ np.diag(chi_mode) @ eta_mat.T

    chi_no_diag = chi_matrix * (1 + np.diag(np.ones(n_ions)*np.nan))
    chi_average = np.sum(np.tril(chi_no_diag, k=-1))/(n_ions*(n_ions-1)/2)
    infidelity = np.sum(np.tril((chi_no_diag/chi_average - 1)*np.pi/4, k=-1)**2)

    inf_arr[i] = infidelity
    omega_arr[i] = i/100

plt.plot(omega_arr, inf_arr)      
plt.xlabel('$\\Omega_{3} / \\Omega_{2}$')
plt.ylabel('GHZ gate infidelity')
plt.grid()
plt.show()

# tone_list = [[Omega0, 0, 0, 0],[Omega0, Omega1, 0, 0],[Omega0, Omega1, Omega2, 0],[Omega0, Omega1, Omega2, Omega3]]
# infidelity_list = np.zeros(4)
# for i in range(0,4):
#     mu_Omega_psi_list = [[mu0, tone_list[i][0], 0], [mu1, tone_list[i][1], 0], [mu2, tone_list[i][2], 0], [mu3, tone_list[i][3], 0]]
#     chi_mode = multitone_chi_mode(t1, t2, t_bp, mu_Omega_psi_list, chain.omegas_ax, omega_mat)
#     chi_matrix = eta_mat @ np.diag(chi_mode) @ eta_mat.T
#     chi_no_diag = chi_matrix * (1 + np.diag(np.ones(n_ions)*np.nan))
#     chi_average = np.sum(np.tril(chi_no_diag, k=-1))/(n_ions*(n_ions-1)/2)
#     infidelity = np.sum(np.tril((chi_no_diag/chi_average - 1)*np.pi/4, k=-1)**2)
#     infidelity_list[i]=infidelity
# plt.plot(np.array([1, 2, 3, 4]), infidelity_list, 's', markersize=5)
# popt, pcov = curve_fit(func, np.array([1, 2, 3, 4]), infidelity_list)
# plt.plot(np.linspace(1, 4, 50), func(np.linspace(1, 4, 50), *popt), '--')
# plt.xlabel('Number of tones')
# plt.ylabel('GHZ gate infidelity')
# plt.title(f'$n_\\mathrm{{ions}} = {n_ions}$')
# plt.grid()
# plt.show()