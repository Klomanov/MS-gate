import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from constants import Constants as c
import shape_param

import shape_calc


class TwoIonsGateDynamics:
    """
    Моделирование динамики двух ионов для реализации квантового гейта
    """

    def __init__(self, m, q, omega_0, omega_t, eta, delta_t, n, params, t_max):
        """
        Инициализация параметров системы

        Parameters:
        m: количество атомов в цепи
        q: количество освещаемых кубитов
        omega_0: частота перехода между уровнями
        omega_t: частоты колебательных мод
        eta: параметры Ламба-Дике
        delta_t: отстройка от боковых полос
        N: число фоковских состояний
        t_max: конечное время гейта
        params: параметры частоты Раби
        """
        self.m = m
        self.q = q
        self.omega_0 = omega_0
        self.omega_t = omega_t
        self.delta_t = delta_t
        self.n = n
        self.eta = eta
        self.params = params

        # Расчет частоты лазеров
        self.omega_l1 = omega_0 + omega_t[-1] + delta_t
        self.omega_l2 = omega_0 - omega_t[-1] - delta_t

        # Время моделирования
        self.t_max = t_max
        self.times = np.linspace(0, self.t_max, 1500)

        # Раби-частота для одной моды
        self.rabi_omega = lambda t: 2 * np.pi / eta[0, 0] / t_max

        # Раби-частота из уравнений
        #self.rabi_omega = lambda t: shape_param.calc(t, params)

        # Инициализация операторов
        self._initialize_operators()

        # Начальное состояние
        self.psi0 = self._initial_state()

        # Результаты моделирования
        self.result = None
        self.populations = {}
        self.phonon_numbers = {}

    def _initialize_operators(self):
        """Инициализация квантовых операторов"""
        # Базовые операторы для подсистем
        spin_ops = [qeye(2) for _ in range(self.q)]
        phonon_ops = [qeye(self.n) for _ in range(self.m)]

        # Операторы Паули для каждого иона
        self.sigma_z = []
        self.sigma_p = []
        self.sigma_m = []
        self.sigma_x = []
        self.sigma_y = []

        for i in range(self.q):
            # Создаем список операторов для тензорного произведения
            ops_list = spin_ops.copy()
            ops_list[i] = sigmaz()
            self.sigma_z.append(tensor(ops_list + phonon_ops))

            ops_list[i] = sigmap()
            self.sigma_p.append(tensor(ops_list + phonon_ops))

            ops_list[i] = sigmam()
            self.sigma_m.append(tensor(ops_list + phonon_ops))

            ops_list[i] = sigmax()
            self.sigma_x.append(tensor(ops_list + phonon_ops))

            ops_list[i] = sigmay()
            self.sigma_y.append(tensor(ops_list + phonon_ops))

        # Операторы колебательных мод
        self.a = []
        self.a_dag = []
        for i in range(self.m):
            ops_list = spin_ops + phonon_ops.copy()
            ops_list[self.q + i] = destroy(self.n)
            self.a.append(tensor(ops_list))

            ops_list[self.q + i] = create(self.n)
            self.a_dag.append(tensor(ops_list))

    def _initial_state(self):
        """Начальное состояние: все ионы в |0>, 0 фононов"""
        spin_states = [basis(2, 0) for _ in range(self.q)]
        phonon_states = [basis(self.n, 0) for _ in range(self.m)]
        return tensor(spin_states + phonon_states)

    def target_bell_state(self):
        """Целевое белловское состояние (|00⟩ - i|11⟩)/√2"""
        state00_ops = [basis(2, 0) for _ in range(self.q)] + [basis(self.n, 0) for _ in range(self.m)]
        state11_ops = [basis(2, 1) for _ in range(self.q)] + [basis(self.n, 0) for _ in range(self.m)]

        state00 = tensor(state00_ops)
        state11 = tensor(state11_ops)

        return (state00 - 1j * state11) / np.sqrt(2)

    def hamiltonian(self, t, args=None):
        """
        Гамильтониан системы в картине взаимодействия

        Returns:
        H: оператор Гамильтониана в момент времени t
        """

        H0 = 0
        for i in range(self.q):
            H0 += self.rabi_omega(t) * np.cos((self.omega_l1 - self.omega_0) * t) * self.sigma_y[i]

        H1 = 0
        for j in range(self.m):
            mode_interaction = (
                    self.a[j] * np.exp(-1j * self.omega_t[j] * t) +
                    self.a_dag[j] * np.exp(1j * self.omega_t[j] * t)
            )
            for i in range(self.q):
                H1 += (self.eta[i, j] * self.rabi_omega(t) * np.cos((self.omega_l1 - self.omega_0) * t) *
                       mode_interaction * self.sigma_x[i])

        return H0 + H1

    def simulate(self):
        """Проведение моделирования динамики системы"""
        print("Запуск моделирования...")
        print(f"Параметры: Omega={self.params}, t_max={self.t_max:.3f}")

        self.result = sesolve(self.hamiltonian, self.psi0, self.times,
                              options=Options(store_states=True))

        self._calculate_observables()

        return self.result

    def _calculate_observables(self):
        """Вычисление наблюдаемых величин"""
        if self.result is None:
            raise ValueError("Сначала выполните simulate()")

        # Базовые состояния для населенностей q=2
        states = {}
        for spins in ['00', '01', '10', '11']:
            spin_ops = []
            for s in spins:
                spin_ops.append(basis(2, int(s)))
            phonon_ops = [basis(self.n, 0) for _ in range(self.m)]
            states[spins] = tensor(spin_ops + phonon_ops)

        # Населенности
        self.populations = {}
        for label, state in states.items():
            self.populations[label] = [np.abs(result_state.overlap(state)) ** 2
                                       for result_state in self.result.states]

        # Числа фононов
        self.phonon_numbers = {}
        for i in range(self.m):
            self.phonon_numbers[f'mode_{i}'] = expect(self.a_dag[i] * self.a[i],
                                                      self.result.states)

    def calculate_fidelity(self, target_time_ratio=0.5):
        """Вычисление фиделитета с целевым состоянием"""
        if self.result is None:
            raise ValueError("Сначала выполните simulate()")

        target_state = self.target_bell_state()
        idx = int(len(self.result.states) * target_time_ratio)

        fid = fidelity(self.result.states[idx], target_state)
        print(f"Фиделити в момент t={self.times[idx]:.3f}: {fid:.6f}")
        return fid

    def plot_results(self):
        """Визуализация результатов"""
        if self.result is None:
            raise ValueError("Сначала выполните simulate()")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Населенности
        for label, pop in self.populations.items():
            ax1.plot(self.times, pop, label=label, linewidth=2)

        ax1.set_xlabel("Время")
        ax1.set_ylabel("Населенность")
        ax1.set_title("Динамика населенностей состояний")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Числа фононов
        for label, phonons in self.phonon_numbers.items():
            ax2.plot(self.times, phonons, label=label, linewidth=2)

        ax2.set_xlabel("Время")
        ax2.set_ylabel("Число фононов")
        ax2.set_title("Динамика чисел фононов")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_parameters(self):
        """Вывод параметров системы"""
        print("Параметры системы:")
        print(f"Количество кубитов в гейте: {self.q}")
        print(f"Количество мод: {self.m}")
        print(f"Частота перехода: {self.omega_0 / (2 * np.pi):.1f} МГц")
        print(f"Частоты мод: {[f'{w / (2 * np.pi):.1f}' for w in self.omega_t]} МГц")
        print(f"Параметры Ламба-Дике: {self.eta}")
        print(f"Отстройка: {self.delta_t / (2 * np.pi):.1f} МГц")
        print(f"Время моделирования: {self.t_max:.3f} мкс")


# Пример использования
if __name__ == "__main__":
    # Создание и настройка системы
    system = TwoIonsGateDynamics(m=c.m, q=c.q, omega_0=c.omega_0, omega_t=c.omega_t,
                                 eta=c.eta, delta_t=c.delta_t, n=c.n, params=c.rabi_omega_params, t_max=c.t_max)

    # Вывод параметров
    system.print_parameters()

    # Проведение моделирования
    result = system.simulate()

    # Вычисление фиделити
    fidelity = system.calculate_fidelity()

    # Визуализация результатов
    system.plot_results()

    # Дополнительный анализ
    print(f"\nМаксимальная населенность |11>: {max(system.populations['11']):.4f}")
    print(f"Максимальное число фононов (мода 0): {max(system.phonon_numbers['mode_0']):.4f}")
    print(f"Максимальное число фононов (мода 1): {max(system.phonon_numbers['mode_1']):.4f}")
