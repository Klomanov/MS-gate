import numpy as np
from scipy.integrate import simpson, cumulative_simpson
import matplotlib.pyplot as plt
from timslib.ion_crystals import IonChain
from scipy.interpolate import PPoly
from scipy.optimize import minimize
from timslib.ms_pulse_shaping.ms_handler import MSHandlerPPolyEqF as MSHandler
import time


class GHZPolyCarrierErrorHandler:
    def __init__(self, t_gate, sm_deg, nu_rad, nu_ax, n_ions, poly_modes):
        """
        Инициализация класса

        Parameters:
        t_gate: время гейта
        nu_rad: радиальная частота ловушки
        nu_ax: аксиальная частота ловушки
        n_ions: число ионов в цепи
        poly_modes: число бихроматических пучков
        """
        self.t_gate = t_gate
        self.sm_deg = sm_deg
        self.nu_rad = nu_rad
        self.nu_ax = nu_ax
        self.n_ions = n_ions
        self.poly_mods = poly_modes

        self.ion_chain = IonChain(nu_rad=nu_rad, nu_ax=nu_ax, n_ions=n_ions, ion_type='Ca40')
        self.omegas = self.ion_chain.omegas_ax
        self.eta = self.ion_chain.eta_ax(0)

        # Omega0 = gamma * np.pi / (self.eta[0, 0] * t_gate)
        # Omega1 = 0.29 * Omega0
        # Omega2 = 0.75 * Omega1
        # Omega3 = 0.78 * Omega2
        # Omega4 = 0.79 * Omega3

        self.delta, self.t_bp, self.omega_mat = self.smoothed_step_mat_from_t()

        # mu0 = 2 * np.pi * (nu_ax - self.delta)
        # mu1 = self.omegas[1] + 2 * np.pi * self.delta
        # mu2 = self.omegas[2] + 2 * np.pi * self.delta
        # mu3 = self.omegas[3] + 2 * np.pi * self.delta
        # mu4 = self.omegas[4] + 2 * np.pi * self.delta
        #
        # self.mu_Omega_psi_list = np.array([[mu0, Omega0, 0], [mu1, Omega1, 0], [mu2, Omega2, 0],
        #                                    [mu3, Omega3, 0], [mu4, Omega4, 0]])
        self.mu_Omega_psi_list = np.zeros((self.poly_mods, 3))

        self.omega_func = PPoly.construct_fast(self.omega_mat, self.t_bp)

        self.n_steps = 2000
        self.t_seg = np.array([self.t_gate / self.n_steps * n for n in range(self.n_steps)])
        self.alpha_tilde_arr = np.zeros((self.n_steps, len(self.omegas)), dtype=complex)
        self.chi_tilde_arr = np.zeros(len(self.omegas))
        self.carrier_phase_arr = np.zeros(self.n_steps)
        self.f_tilde_arr = np.zeros((self.n_steps, len(self.omegas)), dtype=complex)

    def smoothed_step_mat_from_t(self):
        def smoothed_step_mat(t_mid, tau):
            t_bp = np.array([0, tau, t_mid + tau, t_mid + 2 * tau])
            omega_mat = np.array(
                [[-2 / tau ** 3, 3 / tau ** 2, 0, 0], [0, 0, 0, 1], [2 / tau ** 3, -3 / tau ** 2, 0, 1]]).T

            return t_bp, omega_mat

        tau = self.sm_deg * self.t_gate
        t_mid = self.t_gate - 2 * tau
        t_bp, omega_mat = smoothed_step_mat(t_mid, tau)
        delta = 1 / (self.t_gate * (1 - self.sm_deg))

        return delta, t_bp, omega_mat

    def omega_at_t(self, Omega, t):
        return self.omega_func(t) * Omega

    def calc_carrier_phase_arr(self):
        result = 0
        for mu, Omega, psi in self.mu_Omega_psi_list:
            y = np.array([self.omega_at_t(Omega, t) * np.cos(mu * t + psi) for t in self.t_seg])
            result += cumulative_simpson(x=self.t_seg, y=y, initial=0)
        self.carrier_phase_arr = result

    def calc_f_tilde_arr(self):
        for n in range(len(self.t_seg)):
            result = 0
            t = self.t_seg[n]
            for mu, Omega, psi in self.mu_Omega_psi_list:
                result += (np.exp(1j * self.omegas * t) * np.cos(mu * t + psi) *
                           np.cos(2 * self.carrier_phase_arr[n]) * self.omega_at_t(Omega, t))
            self.f_tilde_arr[n] = result

    def calc_alpha_tilde_arr(self):

        y = np.array([-1j * self.f_tilde_arr[n] for n in range(self.n_steps)])

        self.alpha_tilde_arr = (cumulative_simpson(y=np.real(y), x=self.t_seg, axis=0, initial=0) +
                                1j * cumulative_simpson(y=np.imag(y), x=self.t_seg, axis=0, initial=0))

    def calc_chi_tilde_arr(self):
        y = np.array([np.real(self.alpha_tilde_arr[n] * np.conj(self.f_tilde_arr[n])) for n in range(self.n_steps)])
        self.chi_tilde_arr = 2 * np.real(simpson(y=y, x=self.t_seg, axis=0))

    def calc(self):
        self.calc_carrier_phase_arr()
        self.calc_f_tilde_arr()
        self.calc_alpha_tilde_arr()
        self.calc_chi_tilde_arr()

    def infidelity(self, params):
        self.mu_Omega_psi_list[0][1] = params[0]
        self.mu_Omega_psi_list[1][1] = params[1]
        self.mu_Omega_psi_list[0][0] = params[2]
        self.mu_Omega_psi_list[1][0] = params[3]
        self.calc()

        alpha = self.eta * self.alpha_tilde_arr[-1]

        # shaping_attrs = {
        #     "used_ions": [0, 1, 2, 3, 4, 5, 6, 7],
        #     "anglebypi": 0,
        #     "mode_type": "axial",
        #     "psibypi": 0.,
        #     "muby2pi": params[2]/(2*np.pi),
        #     "used_modes" : [0]
        # }
        #
        # handler = MSHandler.get_from_chain(self.ion_chain, self.t_bp, self.omega_mat * params[0], **shaping_attrs)
        # handler.calculate_num_alpha_and_chi_w_carrier(2001)
        #
        # alpha = handler.alpha_fin_num_w_carrier

        chi_matrix = self.eta @ np.diag(self.chi_tilde_arr) @ self.eta.T
        # chi_matrix = handler.chi_fin_num_w_carrier

        infidelity = np.sum(np.abs(alpha) ** 2) + np.sum(np.tril(np.abs(chi_matrix) - np.pi / 4, k=-1) ** 2)

        return infidelity


n_ions = 8
data = np.load('short_ghz_optimal_pulse_params_eta_0_05.npz')

nu_rad = 1.776e6
nu_ax = 0.469e6
t_gate_arr = data['nu_t_gate_arr']/nu_ax
t_gate_arr = t_gate_arr[::5]
beta_arr = data['beta_arr'][::5]
gamma_arr = data['gamma_arr'][::5]

sm_deg = 1/3

inf_arr = np.zeros(len(t_gate_arr))
one_tone_inf_arr = np.zeros(len(t_gate_arr))

Omega0_arr = np.zeros(len(t_gate_arr))
Omega1_arr = np.zeros(len(t_gate_arr))
mu0_arr = np.zeros(len(t_gate_arr))
mu1_arr = np.zeros(len(t_gate_arr))

gamma0_arr = np.zeros(len(t_gate_arr))
gamma1_arr = np.zeros(len(t_gate_arr))
beta0_arr = np.zeros(len(t_gate_arr))
beta1_arr = np.zeros(len(t_gate_arr))

t_gate = t_gate_arr[6]

handler = GHZPolyCarrierErrorHandler(t_gate, sm_deg, nu_rad, nu_ax, n_ions, 2)
Omega0 = gamma_arr[6] * np.pi / (handler.eta[0, 0] * t_gate)
Omega1 = 0.3*Omega0
delta = 2*np.pi*beta_arr[6]/t_gate
mu0 = 2 * np.pi * nu_ax + delta
mu1 = handler.omegas[1] + delta

resolution = 20
Omega0_arr = np.linspace(0.8 * Omega0, 1.2 * Omega0, resolution)
Omega1_arr = np.linspace(0 * Omega1, 1.2 * Omega1, resolution)
mesh_infidelity = np.zeros((resolution, resolution))


for i, om0 in enumerate(Omega0_arr):
    print(i+1)
    for j,  om1 in enumerate(Omega1_arr):
        mesh_infidelity[i, j] = handler.infidelity([om0, om1, mu0, mu1])

Omega0_grid, Omega1_grid = np.meshgrid(Omega0_arr, Omega1_arr, indexing='ij')
im = plt.pcolormesh(Omega0_grid, Omega1_grid, np.log10(mesh_infidelity),
                               shading='auto', cmap='viridis', vmin=-3, vmax=0)

plt.xlabel('$\Omega_0$')
plt.ylabel('$\Omega_1$')
plt.title(f'log10 inf')
plt.colorbar(im, label='log10 inf')

plt.show()

# for i, t_gate in enumerate(t_gate_arr):
#     handler = GHZPolyCarrierErrorHandler(t_gate, sm_deg, nu_rad, nu_ax, n_ions, 2)
#
#     Omega0 = gamma_arr[i] * np.pi / (handler.eta[0, 0] * t_gate)
#     Omega1 = 0.3 * Omega0
#
#     delta = 2*np.pi*beta_arr[i]/t_gate
#
#     mu0 = 2 * np.pi * nu_ax + delta
#     mu1 = handler.omegas[1] + delta
#
#     initial_guess = np.array([Omega0, Omega1, mu0, mu1])
#     bounds = np.array([(Omega0 * 0.8, Omega0 * 1.2), (Omega1 * 0, Omega1 * 1.2),
#                        (mu0 * 0.8, mu0 * 1.2), (mu1 * 0, mu1 * 1.2)])
#
#     result = minimize(handler.infidelity, initial_guess, bounds=bounds)
#     inf_arr[i] = result.fun
#     Omega0_arr[i] = result.x[0]
#     Omega1_arr[i] = result.x[1]
#     mu0_arr[i] = result.x[2]
#     mu1_arr[i] = result.x[3]
#
#     gamma0_arr[i] = Omega0_arr[i] * t_gate * handler.eta[0, 0] / np.pi
#     gamma1_arr[i] = Omega1_arr[i] * t_gate * handler.eta[0, 0] / np.pi
#     beta0_arr[i] = (mu0_arr[i] - handler.omegas[0]) * t_gate/(2 * np.pi)
#     beta1_arr[i] = (mu1_arr[i] - handler.omegas[1]) * t_gate/(2 * np.pi)
#
#     one_tone_inf_arr[i] = handler.infidelity([Omega0, 0, mu0, 0])
#
#     print("Percent: ", np.around((i+1)/np.size(t_gate_arr)*100, 2), ", 1-F:", result.fun)
#
# fig, axes = plt.subplots(2, 3, figsize=(8, 15))
#
# axes[0, 0].plot(t_gate_arr*1e6, gamma0_arr, label="Opt")
# axes[0, 0].plot(t_gate_arr*1e6, gamma_arr, color="black", linestyle="--", label="Initial")
# axes[0, 0].set_ylabel("$\gamma_0$")
# axes[0, 0].set_xlabel("$t$, мкс")
# axes[0, 0].legend()
# axes[0, 0].grid()
#
# axes[0, 1].plot(t_gate_arr*1e6, gamma1_arr)
# axes[0, 1].set_ylabel("$\gamma_1$")
# axes[0, 1].set_xlabel("$t$, мкс")
# axes[0, 1].grid()
#
# axes[0, 2].plot(t_gate_arr*1e6, beta0_arr, label="Opt")
# axes[0, 2].plot(t_gate_arr*1e6, beta_arr, color="black", linestyle="--", label="Initial")
# axes[0, 2].set_ylabel("$\\beta_0$")
# axes[0, 2].set_xlabel("$t$, мкс")
# axes[0, 2].legend()
# axes[0, 2].grid()
#
# axes[1, 0].plot(t_gate_arr*1e6, beta1_arr)
# axes[1, 0].set_ylabel("$\\beta_1$")
# axes[1, 0].set_xlabel("$t$, мкс")
# axes[1, 0].grid()
#
# axes[1, 1].plot(t_gate_arr*1e6, np.log10(inf_arr), label="Multi tons")
# axes[1, 1].plot(t_gate_arr*1e6, np.log10(one_tone_inf_arr), color="black", linestyle="--", label="One ton")
# axes[1, 1].set_ylabel("$\log_{10} 1-F$")
# axes[1, 1].set_xlabel("$t$, мкс")
# axes[1, 1].legend()
# axes[1, 1].grid()
#
# axes[1, 2].set_visible(False)
#
# axes[1, 1].set_ylim(bottom=-3, top=0)
#
# plt.grid()
# plt.show()


