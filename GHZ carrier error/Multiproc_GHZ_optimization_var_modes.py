import numpy as np
from scipy.integrate import simpson, cumulative_simpson
import matplotlib.pyplot as plt
from timslib.ion_crystals import IonChain
from scipy.interpolate import PPoly
from scipy.optimize import minimize
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from typing import List, Tuple
from scipy.signal import savgol_filter


@dataclass
class LaserData:
    n_modes: int
    mu: np.ndarray
    Omega: np.ndarray
    psi: np.ndarray


class GHZInfidelityCalculator:
    def __init__(self, t_gate: float, sm_deg: float, nu_rad: float, nu_ax: float, n_ions: int,
                 n_active_modes: int = None):
        self.t_gate = t_gate
        self.n_ions = n_ions
        self.ion_chain = IonChain(nu_rad=nu_rad, nu_ax=nu_ax, n_ions=n_ions, ion_type='Ca40')

        all_omegas = self.ion_chain.omegas_ax
        all_etas = self.ion_chain.eta_ax(0)

        if n_active_modes is not None:
            if n_active_modes > len(all_omegas):
                raise ValueError(f"Запрошено {n_active_modes} мод, но доступно только {len(all_omegas)}")
            self.omegas_ax = all_omegas[:n_active_modes]
            self.eta_im = all_etas[:, :n_active_modes]
        else:
            self.omegas_ax = all_omegas
            self.eta_im = all_etas

        self.t_bp, self.omega_mat = self._smoothed_step_mat_from_t(sm_deg)
        self.omega_func = PPoly.construct_fast(self.omega_mat, self.t_bp)

        dt_target = 0.02e-6
        self.n_steps = int(self.t_gate / dt_target)

        if self.n_steps < 1000:
            self.n_steps = 1000

        self.t_seg = np.linspace(0, self.t_gate, self.n_steps)
        self.pulse_shape_t = self.omega_func(self.t_seg)
        self.laser_data = None

    def _smoothed_step_mat_from_t(self, sm_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        tau = sm_deg * self.t_gate
        t_mid = self.t_gate - 2 * tau
        t_bp = np.array([0, tau, t_mid + tau, self.t_gate])
        omega_mat = np.array([[-2 / tau ** 3, 3 / tau ** 2, 0, 0],
                              [0, 0, 0, 1],
                              [2 / tau ** 3, -3 / tau ** 2, 0, 1]]).T
        return t_bp, omega_mat

    def calculate_infidelity(self, gammas: List[float], betas: List[float], phis: List[float]) -> float:

        eta_com = self.eta_im[0, 0]
        n_tones = len(gammas)

        deltas = 2 * np.pi * np.array(betas) / self.t_gate

        mu = np.array([self.omegas_ax[i] + deltas[i] for i in range(n_tones)])

        Omega = np.array(gammas) * np.pi / (eta_com * self.t_gate)

        self.laser_data = LaserData(n_modes=n_tones, mu=mu, Omega=Omega, psi=np.array(phis))

        carrier_phase = self._calculate_carrier_phase()
        f_tilde = self._calculate_f_tilde(carrier_phase)
        alpha_tilde = self._calculate_alpha_tilde(f_tilde)
        chi_tilde = self._calculate_chi_tilde(f_tilde, alpha_tilde)

        alpha_final = self.eta_im @ alpha_tilde[-1, :]
        infidelity_displacement = np.sum(np.abs(alpha_final) ** 2)

        chi_matrix = self.eta_im @ np.diag(chi_tilde) @ self.eta_im.T

        infidelity_phase = np.sum(np.tril(np.abs(chi_matrix) - np.pi / 4, k=-1) ** 2)

        return infidelity_displacement + infidelity_phase

    def _calculate_carrier_phase(self) -> np.ndarray:
        mu_t = self.laser_data.mu[:, np.newaxis] * self.t_seg + self.laser_data.psi[:, np.newaxis]
        integrand = self.laser_data.Omega[:, np.newaxis] * self.pulse_shape_t * np.cos(mu_t)
        total_integrand = np.sum(integrand, axis=0)
        return cumulative_simpson(y=total_integrand, x=self.t_seg, initial=0)

    def _calculate_f_tilde(self, carrier_phase: np.ndarray) -> np.ndarray:
        mu_t = self.laser_data.mu[:, np.newaxis] * self.t_seg + self.laser_data.psi[:, np.newaxis]
        omega_m_t = self.omegas_ax[np.newaxis, :, np.newaxis] * self.t_seg

        term1 = np.cos(mu_t)[:, np.newaxis, :]
        term2 = np.exp(1j * omega_m_t)

        summand = self.laser_data.Omega[:, np.newaxis, np.newaxis] * term1 * term2

        f_tilde = np.sum(summand, axis=0).T
        f_tilde *= self.pulse_shape_t[:, np.newaxis] * np.cos(2 * carrier_phase)[:, np.newaxis]
        return f_tilde

    def _calculate_alpha_tilde(self, f_tilde: np.ndarray) -> np.ndarray:
        y = -1j * f_tilde
        integral_real = cumulative_simpson(y=np.real(y), x=self.t_seg, initial=0, axis=0)
        integral_imag = cumulative_simpson(y=np.imag(y), x=self.t_seg, initial=0, axis=0)
        return integral_real + 1j * integral_imag

    def _calculate_chi_tilde(self, f_tilde: np.ndarray, alpha_tilde: np.ndarray) -> np.ndarray:
        integrand = np.real(alpha_tilde * np.conj(f_tilde))
        return 2 * np.real(simpson(y=integrand, x=self.t_seg, axis=0))


def objective_function(params: np.ndarray, handler: GHZInfidelityCalculator) -> float:
    n_tones = len(params) // 2
    gammas = params[:n_tones]
    betas = params[n_tones:]
    phis = [0.0] * n_tones
    return handler.calculate_infidelity(gammas, betas, phis)


def process_chunk(args_pack):
    (chunk_indices, t_gate_subset,
     gamma_file_subset, beta_file_subset,
     consts) = args_pack

    n_ions, sm_deg, nu_rad, nu_ax, n_active_modes = consts

    results = []
    current_optimal_params = None

    for i in range(len(t_gate_subset)):
        t_gate = t_gate_subset[i]
        handler = GHZInfidelityCalculator(t_gate, sm_deg, nu_rad, nu_ax, n_ions, n_active_modes)

        init_gammas_file = [gamma_file_subset[i], 0.3 * gamma_file_subset[i]]
        init_betas_file = [beta_file_subset[i], beta_file_subset[i]]
        guess_from_file = np.concatenate([init_gammas_file, init_betas_file])

        if current_optimal_params is None:
            actual_guess = guess_from_file
        else:
            actual_guess = current_optimal_params.copy()
            if actual_guess[1] < 0.02:
                actual_guess[1] = guess_from_file[1]
                actual_guess[3] = guess_from_file[3]

        bounds = ((0.0, np.inf), (0.0, np.inf), (None, None), (None, None))

        res = minimize(
            objective_function,
            actual_guess,
            args=(handler,),
            method='L-BFGS-B',
            bounds=bounds,
            tol=1e-10
        )

        if res.fun > 1e-1:
            res_rescue = minimize(
                objective_function,
                guess_from_file,
                args=(handler,),
                method='L-BFGS-B',
                bounds=bounds,
                tol=1e-10
            )
            if res_rescue.fun < res.fun:
                res = res_rescue

        current_optimal_params = res.x

        one_tone_gammas = [res.x[0], 0.0]
        one_tone_betas = [res.x[2], 0.0]
        one_inf = handler.calculate_infidelity(one_tone_gammas, one_tone_betas, [0.0, 0.0])

        res_pack = {
            'global_index': chunk_indices[i],
            'x': res.x,
            'fun': res.fun,
            'one_inf': one_inf
        }
        results.append(res_pack)

    return results


def main():
    n_ions = 4
    nu_rad = 1.9e6
    nu_ax = 0.93e6
    sm_deg = 1 / 3
    resol = 5
    N_modes_to_consider = 2

    data_path = 'short_ghz_optimal_pulse_params_eta_0_05.npz'
    save_path = 'two_tone_ghz_results.npz'

    data = np.load(data_path)
    t_gate_arr_full = data['nu_t_gate_arr'] / nu_ax
    gamma_init_arr_full = data['gamma_arr']
    beta_init_arr_full = data['beta_arr']

    slice_ = slice(0, None, resol)
    t_gate_arr = t_gate_arr_full[slice_]
    gamma_from_file = gamma_init_arr_full[slice_]
    beta_from_file = beta_init_arr_full[slice_]

    n_cores = cpu_count()
    total_points = len(t_gate_arr)

    indices_split = np.array_split(np.arange(total_points), n_cores)
    t_gate_split = np.array_split(t_gate_arr, n_cores)
    gamma_file_split = np.array_split(gamma_from_file, n_cores)
    beta_file_split = np.array_split(beta_from_file, n_cores)

    consts = (n_ions, sm_deg, nu_rad, nu_ax, N_modes_to_consider)

    tasks = []
    for i in range(len(indices_split)):
        if len(indices_split[i]) == 0: continue
        task = (
            indices_split[i],
            t_gate_split[i],
            gamma_file_split[i],
            beta_file_split[i],
            consts
        )
        tasks.append(task)

    print(f"Запуск оптимизации: {total_points} точек на {len(tasks)} ядрах")
    if N_modes_to_consider:
        print(f"Учет первых {N_modes_to_consider} мод")

    start_time = time.time()

    final_results = []
    with Pool(processes=len(tasks)) as p:
        for chunk_res in tqdm(p.imap_unordered(process_chunk, tasks), total=len(tasks)):
            final_results.extend(chunk_res)

    final_results.sort(key=lambda r: r['global_index'])

    end_time = time.time()
    print(f"Завершено за {end_time - start_time:.2f} секунд")

    results_gamma = [r['x'][:2] for r in final_results]
    results_beta = [r['x'][2:] for r in final_results]
    inf_arr = np.array([r['fun'] for r in final_results])
    one_tone_inf_arr = np.array([r['one_inf'] for r in final_results])

    gamma_arrs = np.array(results_gamma).T
    beta_arrs = np.array(results_beta).T

    np.savez(
        save_path,
        n_ions=n_ions,
        nu_ax=nu_ax,
        nu_rad=nu_rad,
        sm_deg=sm_deg,
        gamma_arrs=gamma_arrs,
        beta_arrs=beta_arrs,
        t_gate_arr=t_gate_arr,
        inf_arr=inf_arr,
        inf_arr_one=one_tone_inf_arr
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), tight_layout=True)
    axes = axes.ravel()

    axes[0].plot(t_gate_arr * 1e6, gamma_arrs[0], label=r"Opt $\gamma_0$")
    axes[0].plot(t_gate_arr * 1e6, gamma_from_file, 'k--', label=r"Initial $\gamma_0$")
    axes[0].set_ylabel(r"$\gamma_0$")
    axes[0].grid(True)

    axes[1].plot(t_gate_arr * 1e6, gamma_arrs[1], label=r"Opt $\gamma_1$")
    axes[1].set_ylim((0, 1))
    axes[1].set_ylabel(r"$\gamma_1$")
    axes[1].grid(True)

    axes[2].plot(t_gate_arr * 1e6, beta_arrs[0], label=r"Opt $\beta_0$")
    axes[2].plot(t_gate_arr * 1e6, beta_from_file, 'k--', label=r"Initial $\beta_0$")
    axes[2].set_ylabel(r"$\beta_0$")
    axes[2].grid(True)

    axes[3].plot(t_gate_arr * 1e6, beta_arrs[1], label=r"Opt $\beta_1$")
    axes[3].set_ylabel(r"$\beta_1$")
    axes[3].grid(True)

    safe_inf = np.maximum(inf_arr, 1e-16)
    safe_one = np.maximum(one_tone_inf_arr, 1e-16)

    axes[4].plot(t_gate_arr * 1e6, np.log10(safe_inf), label="Multi-tone")
    axes[4].plot(t_gate_arr * 1e6, np.log10(safe_one), 'k--', label="Single-tone")
    axes[4].set_ylabel(r"$\log_{10} (1-F)$")
    axes[4].set_ylim(-8, 0)
    axes[4].legend()
    axes[4].grid(True)

    axes[5].set_visible(False)

    for ax in axes[:5]:
        ax.set_xlabel(r"$t_{gate}$, мкс")

    plt.show()


if __name__ == '__main__':
    main()
