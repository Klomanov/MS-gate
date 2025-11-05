import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.interpolate import CubicSpline
from timslib.ms_pulse_shaping.ms_handler import MSHandlerPPolyEqF as MSHandler


class FidelityCalculator:
    def __init__(self, data, n_ions):
        self.n_ions = n_ions
        self.n_ions_arr = data['n_ions_arr']
        self.nu_ax_arr = data['nu_ax_arr']
        self.t_gate_arr = data['t_gate_arr']
        self.beta_arr = data['beta_arr']
        self.gamma_arr = data['gamma_arr']
        self.inf_arr = data['inf_arr']
        self.epsilon = 1 / 3
        self.spline_segments = 100
        self.index = np.where(self.n_ions_arr == self.n_ions)[0][0]  # Индекс в массивах для конкретного числа ионов

        self.nu_ax = self.nu_ax_arr[self.index]
        self.eta = np.sqrt(9394 * 1e-6 / self.n_ions / self.nu_ax)

    def rect_s3_mat(self, T, epsilon):
        def smoothstep_mat(t_mid, tau):
            t_bp = np.array([0, tau, t_mid + tau, t_mid + 2 * tau], dtype=np.float64)
            smoothstep_mat = np.array([[-2 / tau ** 3, 3 / tau ** 2, 0, 0],
                                       [0, 0, 0, 1],
                                       [2 / tau ** 3, -3 / tau ** 2, 0, 1]], dtype=np.float64).T
            return t_bp, smoothstep_mat

        t_mid = T * (1 - 2 * epsilon)
        tau = epsilon * T
        t_bp, Omega_mat = smoothstep_mat(t_mid, tau)
        return t_bp, Omega_mat

    def omega(self, t_gate, omega_max, t):
        t_bp, omega_mat = self.rect_s3_mat(t_gate, self.epsilon)
        omega_mat *= omega_max

        t0, t1, t2, t3 = t_bp

        if t < t0:
            return 0.0
        elif t < t1:  # фаза подъема
            dt = t - t0
            a, b, c, d = omega_mat[:, 0]
            return a * dt ** 3 + b * dt ** 2 + c * dt + d
        elif t < t2:  # плоская часть
            a, b, c, d = omega_mat[:, 0]
            return a * t1 ** 3 + b * t1 ** 2 + c * t1 + d
        elif t < t3:  # фаза спада
            dt = t - t2
            a, b, c, d = omega_mat[:, 2]
            return a * dt ** 3 + b * dt ** 2 + c * dt + d
        else:  # после импульса
            return 0.0

    def omega_eff(self, t_gate, omega_max, mu, t):
        omega_t = self.omega(t_gate, omega_max, t)
        j0_term = jv(0, 2 * omega_t / mu)  # J0(2Ω/μ)
        j2_term = jv(2, 2 * omega_t / mu)  # J2(2Ω/μ)

        return omega_t * (j0_term + j2_term)

    def calc(self):
        my_inf_arr = []
        for i in range(len(self.t_gate_arr)):
            t_gate = self.t_gate_arr[i]

            omega_max = np.pi * self.gamma_arr[self.index][i] / self.eta / t_gate
            delta = 2 * np.pi * self.beta_arr[self.index][i] / t_gate
            mu = 2 * np.pi * self.nu_ax + delta

            t_bp = np.linspace(0, t_gate, self.spline_segments + 1)
            omega_eff_values = [self.omega_eff(t_gate, omega_max, mu, i) for i in t_bp]

            spline = CubicSpline(t_bp, omega_eff_values)

            omega_mat = spline.c

            handler = MSHandler(self.eta * np.array([[1], [1]]), np.array([self.nu_ax * 2 * np.pi]), mu, 0, t_bp,
                                omega_mat)
            n_t = 2001
            handler.calculate_num_alpha_and_chi(n_t)

            chi = handler.chi_fin_num[0, 1]
            alpha = handler.alpha_fin_num[1, 0]

            infidelity = self.n_ions * np.abs(alpha) ** 2 + self.n_ions * (self.n_ions - 1) / 2 * (chi - np.pi / 4) ** 2
            my_inf_arr.append(infidelity)

        return my_inf_arr

    def plot(self):
        x = self.t_gate_arr
        new_inf = self.calc()
        old_inf = self.inf_arr[self.index]

        plt.plot(x, new_inf, "r--", label="Bessel error")
        plt.plot(x, old_inf, "b--", label="Old data")
        plt.text(200, 10e-6, f"{self.n_ions} ионов", size=16)
        plt.legend()
        plt.ylabel("$1-F$")
        plt.yscale("log")
        plt.ylim(10 ** (-10), 0)

        plt.show()


if __name__ == "__main__":
    data = np.load('short_ghz_optimal_pulse_params.npz')

    n_ions_arr = data['n_ions_arr']
    nu_ax_arr = data['nu_ax_arr']
    t_gate_arr = data['t_gate_arr']
    beta_arr = data['beta_arr']
    gamma_arr = data['gamma_arr']
    inf_arr = data['inf_arr']

    fig, axes = plt.subplots(6, 2, figsize=(8, 15))

    n_to_ax_dict = {
        2: ((0, 0), (1, 0)),
        4: ((0, 1), (1, 1)),
        6: ((2, 0), (3, 0)),
        8: ((2, 1), (3, 1)),
        10: ((4, 0), (5, 0)),
        12: ((4, 1), (5, 1))
    }

    for n_ions in n_ions_arr:
        calculator = FidelityCalculator(data, n_ions)
        my_inf_arr = calculator.calc()

        t1, t2 = n_to_ax_dict[n_ions]
        ax1 = axes[t1[0], t1[1]]
        ax2 = axes[t2[0], t2[1]]
        ax1.legend()
        i = list(n_ions_arr).index(n_ions)
        ax1.set_title(fr'$\nu_{{ax}} = {nu_ax_arr[i]}MHz, \ n_{{ions}} = {n_ions}$')
        ax1.plot(t_gate_arr, beta_arr[i], label=r'$\beta(t_{gate})$')
        ax1.plot(t_gate_arr, gamma_arr[i], label=r'$\gamma(t_{gate})$')
        ax2.plot(t_gate_arr, inf_arr[i], 'r--', label='$1-F \ Old$')
        ax2.plot(t_gate_arr, my_inf_arr, 'b--', label='$1-F \ Bessel$')
        ax2.set_yscale('log')
        ax2.set_ylim(ymin=1e-10)
        ax2.legend(loc='upper right')
        ax2.set_xlabel('$t_{gate}$ ($\mu$s)')
        ax2.set_ylabel('$1-F$')

    plt.tight_layout()
    plt.show()
