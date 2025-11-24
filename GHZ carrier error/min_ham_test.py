from qutip import tensor, fock, qeye, destroy, jmat, sesolve, fidelity, sigmax, sigmay, QobjEvo, Qobj
import numpy as np
import math
from scipy.interpolate import PPoly
from timslib.ion_crystals import IonChain
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time

def ideal_MS(n_ions, theta):
    sx_list = []
    for i in range(n_ions):
        op_list = [sigmax() if i == j else qeye(2) for j in range(n_ions)]
        sx_list.append(tensor(op_list))
    Sx_total = 0.5 * sum(sx_list)
    return (1j * 2 * theta * Sx_total * Sx_total).expm()

def smoothed_step_mat_from_t(t_gate, sm_deg):
    tau = sm_deg * t_gate
    t_mid = t_gate - 2 * tau
    t_bp = np.array([0, tau, t_mid + tau, t_gate])
    s_mat = np.array([[-2/tau**3, 3/tau**2, 0, 0], [0, 0, 0, 1], [2/tau**3, -3/tau**2, 0, 1]]).T
    return t_bp, s_mat

def create_full_hamiltonian(N_ions, N_cutoff, motional_freqs, t_range, mu_alpha, Omega_alpha_t, psi_alpha, eta_im):
    N_modes = len(motional_freqs)
    hilbert_space = [qeye(2)] * N_ions + [qeye(N_cutoff)] * N_modes

    def get_op(op, index, is_spin_op=True):
        op_list = list(hilbert_space)
        op_list[index if is_spin_op else N_ions + index] = op
        return tensor(op_list)

    sigma_x_list = [get_op(sigmax(), i) for i in range(N_ions)]
    sigma_y_list = [get_op(sigmay(), i) for i in range(N_ions)]
    a_ops = [get_op(destroy(N_cutoff), m, is_spin_op=False) for m in range(N_modes)]
    ad_ops = [op.dag() for op in a_ops]

    ham_list = []
    Sy_total = sum(sigma_y_list)

    for alpha in range(len(mu_alpha)):
        C_alpha_t = Omega_alpha_t[alpha] * np.cos(mu_alpha[alpha] * t_range + psi_alpha[alpha])
        ham_list.append([Sy_total, C_alpha_t])

        for i in range(N_ions):
            for m in range(N_modes):
                eta = eta_im[i, m]
                if abs(eta) < 1e-9: continue
                op1 = eta * sigma_x_list[i] * a_ops[m]
                coeff1 = C_alpha_t * np.exp(-1j * motional_freqs[m] * t_range)
                ham_list.append([op1, coeff1])
                op2 = eta * sigma_x_list[i] * ad_ops[m]
                coeff2 = C_alpha_t * np.exp(1j * motional_freqs[m] * t_range)
                ham_list.append([op2, coeff2])
    return QobjEvo(ham_list, tlist=t_range)

def run_single_simulation_1(args):

    i, t_gate, params, consts = args
    gamma_0, gamma_1, beta_0, beta_1 = params
    N_ions, N_cutoff, N_modes_to_include, sm_deg, full_eta_im, motional_freqs, psi_0, target_spin_state, eta_im, time_resol, solver_steps = consts

    tone_config = [
        {'mode_index': 0, 'delta': 2 * np.pi * beta_0 / t_gate, 'sideband': 'red',
         'Omega_peak': gamma_0 * np.pi / (full_eta_im[0, 0] * t_gate)}
    ]
    psi_alpha = np.zeros(len(tone_config))

    t_bp, s_mat = smoothed_step_mat_from_t(t_gate, sm_deg=sm_deg)
    t_range = np.linspace(0, t_gate, int(t_gate/time_resol) + 1)
    pulse_shape = PPoly.construct_fast(s_mat, t_bp)(t_range)

    mu_alpha = []
    Omega_alpha_t = []
    for tone in tone_config:
        omega_m = motional_freqs[tone['mode_index']]
        delta = tone['delta']
        mu_alpha.append(omega_m + delta)
        Omega_alpha_t.append(tone['Omega_peak'] * pulse_shape)

    H = create_full_hamiltonian(N_ions, N_cutoff, motional_freqs, t_range, mu_alpha, Omega_alpha_t, psi_alpha, eta_im)

    solver_options = {"store_states": False, "store_final_state": True, "nsteps": solver_steps, "atol": 1e-8, "rtol": 1e-6}
    result = sesolve(H, psi_0, t_range, e_ops=[], options=solver_options)
    final_state = result.final_state

    final_spin_state = final_state.ptrace(list(range(N_ions)))
    final_fidelity = fidelity(final_spin_state, target_spin_state)
    infidelity = 1 - final_fidelity**2

    return {'index': i, 'infidelity': infidelity}

def run_single_simulation_2(args):

    i, t_gate, params, consts = args
    gamma_0, gamma_1, beta_0, beta_1 = params
    N_ions, N_cutoff, N_modes_to_include, sm_deg, full_eta_im, motional_freqs, psi_0, target_spin_state, eta_im, time_resol, solver_steps = consts

    tone_config = [
        {'mode_index': 0, 'delta': 2 * np.pi * beta_0 / t_gate, 'sideband': 'red',
         'Omega_peak': gamma_0 * np.pi / (full_eta_im[0, 0] * t_gate)},
        {'mode_index': 1, 'delta': 2 * np.pi * beta_1 / t_gate, 'sideband': 'blue',
         'Omega_peak': gamma_1 * np.pi / (full_eta_im[0, 0] * t_gate)}
    ]
    psi_alpha = np.zeros(len(tone_config))

    t_bp, s_mat = smoothed_step_mat_from_t(t_gate, sm_deg=sm_deg)
    t_range = np.linspace(0, t_gate, int(t_gate/time_resol) + 1)
    pulse_shape = PPoly.construct_fast(s_mat, t_bp)(t_range)

    mu_alpha = []
    Omega_alpha_t = []
    for tone in tone_config:
        omega_m = motional_freqs[tone['mode_index']]
        delta = tone['delta']
        mu_alpha.append(omega_m + delta)
        Omega_alpha_t.append(tone['Omega_peak'] * pulse_shape)

    H = create_full_hamiltonian(N_ions, N_cutoff, motional_freqs, t_range, mu_alpha, Omega_alpha_t, psi_alpha, eta_im)

    solver_options = {"store_states": False, "store_final_state": True, "nsteps": solver_steps, "atol": 1e-8, "rtol": 1e-6}
    result = sesolve(H, psi_0, t_range, e_ops=[], options=solver_options)
    final_state = result.final_state

    final_spin_state = final_state.ptrace(list(range(N_ions)))
    final_fidelity = fidelity(final_spin_state, target_spin_state)
    infidelity = 1 - final_fidelity**2

    return {'index': i, 'infidelity': infidelity}


if __name__ == '__main__':
    start_time = time.time()

    npz_file_path = 'two_tone_ghz_results.npz'
    npz_inf_min_path = "two_tone_ghz_min_results.npz"

    data = np.load(npz_file_path)
    data2 = np.load(npz_inf_min_path)

    t_gate_arr = data['t_gate_arr'][:30:1]
    gamma_arrs = data['gamma_arrs']
    beta_arrs = data['beta_arrs']
    inf_arr_analytical = data['inf_arr'][:30:1]
    inf_arr_one_tone = data['inf_arr_one'][:30:1]

    inf_analytic_min = data2["inf_arr"]
    gamma_analytic_min = data2["gamma1_arr"]

    N_ions = int(data['n_ions'])
    nu_rad = float(data['nu_rad'])
    nu_ax = float(data['nu_ax'])
    sm_deg = float(data['sm_deg'])

    N_modes_to_include = 2
    N_cutoff = 26

    time_resol = 0.01e-6
    solver_steps = 5000

    chain = IonChain(nu_rad=nu_rad, nu_ax=nu_ax, n_ions=N_ions, ion_type='Ca40')
    all_motional_freqs = chain.omegas_ax
    full_eta_im = chain.eta_ax(0)
    motional_freqs = all_motional_freqs[:N_modes_to_include]
    eta_im = full_eta_im[:, :N_modes_to_include]

    spin_initial = tensor([fock(2, 0) for _ in range(N_ions)])
    motion_initial = tensor([fock(N_cutoff, 0) for _ in range(N_modes_to_include)])
    psi_0 = tensor(spin_initial, motion_initial)

    ideal_op = ideal_MS(N_ions, -np.pi / 4)ю.
    target_spin_state = ideal_op * spin_initial

    tasks = []
    constants = (N_ions, N_cutoff, N_modes_to_include, sm_deg, full_eta_im, motional_freqs, psi_0, target_spin_state, eta_im, time_resol, solver_steps)

    ind = 7
    t_gate = t_gate_arr[7]
    gamma_test_arr = np.linspace(gamma_arrs[1, ind]*0.95, gamma_arrs[1, ind]*1.05, 50)

    print(t_gate*1e6, "мкс")
    print("Gamma1:", gamma_arrs[1, ind])
    print("Beta1:", beta_arrs[1, ind])
    print("Gamma0:", gamma_arrs[0, ind])
    print("Beta0:", beta_arrs[0, ind])

    for i in range(len(gamma_test_arr)):
        params = (gamma_arrs[0, ind], gamma_test_arr[i], beta_arrs[0, ind], beta_arrs[1, ind])
        args = (ind, t_gate, params, constants)
        tasks.append(args)

    print(f"Запуск {2*len(tasks)} симуляций на {cpu_count()} ядрах...")

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(run_single_simulation_2, tasks), total=len(tasks)))

    results.sort(key=lambda r: r['index'])
    simulated_infidelity_arr_2 = np.array([r['infidelity'] for r in results])

    end_time = time.time()
    print(f"Полное время: {end_time - start_time:.2f} секунд")

    plt.figure(figsize=(10, 6))

    plt.plot(gamma_test_arr, np.log10(simulated_infidelity_arr_2), 's-', label='Two Tone Simulation Infidelity', markersize=4)
    plt.plot(gamma_analytic_min, np.log10(inf_analytic_min), '-', label='Two Tone Analytical Infidelity')

    plt.title(f'Simulated infidelity comparsion for {N_ions} ions and {N_modes_to_include} modes')
    plt.xlabel(r'$\gamma_1$')
    plt.ylabel(r'$\log_{10} (1-F)$')
    plt.grid(True)
    plt.legend()
    plt.show()