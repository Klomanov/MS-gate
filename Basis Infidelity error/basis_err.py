import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from timslib.ms_pulse_shaping.ms_handler import MSHandlerPPolyEqF as MSHandler
from timslib.ion_crystals.normal_modes import nu_ax_from_nu_rad_min
from timslib.ion_crystals.ion_chain import IonChain
from timslib.ms_pulse_shaping.pulse_shaping import ExceedMaxOmegaError

n_ions = 20

nu_rad = 2.5e6
nu_rad_min = 0.2e6
nu_ax = nu_ax_from_nu_rad_min(nu_rad_min, nu_rad, n_ions)

step = 5
#n_df = np.array([n for n in range(1, 2*n_ions + 1 + step, step)])
n_df = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
t_gate = np.array([t * 1e-6 for t in range(20, 140)])
mu = np.array([nu_rad * (1 + m / 200) for m in range(-20, 20)])
inf = np.zeros((len(n_df), len(t_gate), len(mu)))
Omega_by_mu = np.zeros((len(n_df), len(t_gate), len(mu)))

for n in tqdm(range(len(n_df))):
    for t in range(len(t_gate)):
        for m in range(len(mu)):
            chain = IonChain(nu_rad=nu_rad, nu_ax=nu_ax, n_ions=n_ions, ion_type='Ca40')
            shaping_attrs = {
                "used_ions": [4, 5],
                "anglebypi": 0.5,
                "mode_type": "radial",
                "psibypi": 0.,
                "thetabypi": 0.25,
                "n_df": n_df[n],
                "shaping_type": "3",
                "muby2pi": mu[m],
                "t_gate": t_gate[t],
                "smoothing_degree": 0.3
            }
            try:
                handler = MSHandler.get_from_chain_with_pulse_shaping(chain, carrier_opt=True,
                                                                      **shaping_attrs)
                handler.calculate_num_alpha_and_chi_w_carrier(2001)
                chi = handler.chi_fin_num_w_carrier
                alpha = handler.alpha_fin_num_w_carrier
                infidelity = np.sum(np.abs(alpha)**2) + np.sum(np.abs(np.abs(chi[0, 1]) - np.pi/4)**2)
                inf[n, t, m] = infidelity
                Omega_by_mu[n, t, m] = handler.Omega(t_gate[t]/2)/mu[m]/(2*np.pi)
            except ExceedMaxOmegaError:
                inf[n, t, m] = np.nan
                Omega_by_mu[n, t, m] = np.nan


T_grid, MU_grid = np.meshgrid(t_gate, mu, indexing='ij')

# Определяем количество строк и столбцов
rows = 2
cols = (len(n_df) + rows - 1)//rows


fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

if cols == 1:
    axes = axes.reshape(1, -1)

# Заполняем графики
for n in range(len(n_df)):
    row = n // cols
    col = n % cols

    im = axes[row, col].pcolormesh(MU_grid, T_grid * 1e6, np.log10(inf[n, :, :]),
                                   shading='auto', cmap='viridis', vmin=-5, vmax=0)
    axes[row, col].set_xlabel('mu (Hz)')
    axes[row, col].set_ylabel('t_gate (µs)')
    axes[row, col].set_title(f'log10 inf for n_df = {n_df[n]}')
    plt.colorbar(im, ax=axes[row, col], label='log10 inf')

# Скрываем пустые subplots
if len(n_df) % cols != 0:
    for empty_col in range(len(n_df) % cols, cols):
        axes[rows-1, empty_col].set_visible(False)

#print(np.min(Omega_by_mu))

plt.tight_layout()
plt.show()
