import random
import numpy as np
from scipy.optimize import root
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt
from constants import Constants as c
import shape_param


class GatePulseSolver:
    def __init__(self, t_max, n_params, m, q, eta, omega_t, mu, psi):
        """
        Инициализация решателя для импульса Ω(t)

        Parameters:
        t_max: время гейта
        n_params: количество параметров для параметризации Ω(t)
        m: число мод колебаний
        q: число кубитов
        """

        self.t_max = t_max
        self.n_params = n_params
        self.m = m
        self.q = q

        # Физические параметры (примерные значения)
        self.eta = eta
        self.omega_t = omega_t  # ω_m
        self.mu = mu  # частота модуляции
        self.psi = psi  # фаза модуляции

        self.iterations = 0

    def f_im0(self, t, i, m, params):
        """Функция f_im^0(t) из уравнения (8)"""
        eta_im = self.eta[i, m]
        return eta_im * np.exp(1j * self.omega_t[m] * t) * shape_param.calc(t, params) * np.cos(self.mu * t + self.psi)

    def alpha_im0(self, t2, t1, i, m, params):
        """α_im^0(t2, t1) из уравнения (6)"""

        def integrand(t):
            return self.f_im0(t, i, m, params)

        t_eval = np.linspace(t1, t2, 800)  # Плотная сетка
        integrand_values = [integrand(t) for t in t_eval]

        result = simpson(y=integrand_values, x=t_eval)
        return -1j * result

    def chi_ij0(self, t2, t1, i, j, params):
        """χ_ij^0(t2, t1) из уравнения (7)"""

        def integrand(t):
            total = 0.0
            for k in range(self.m):
                alpha_val_im = self.alpha_im0(t, t1, i, k, params)  # суммируем по m
                f_conj_jm = np.conj(self.f_im0(t, j, k, params))
                total += alpha_val_im * f_conj_jm
            return total

        t_eval = np.linspace(t1, t2, 800)  # Плотная сетка
        integrand_values = [integrand(t) for t in t_eval]

        result = simpson(y=integrand_values, x=t_eval)
        # result, error = quad(lambda t: integrand(t).real, t1, t2)
        return 2 * result.real

    def equations(self, params, target_phi=np.pi / 2):
        """
        Система уравнений, которую нужно решить

        Returns:
        residuals: вектор невязок [α_conditions, χ_condition]
        """

        residuals = []

        # Условие 1: α_im^0(tf, t0) = 0 для всех i, m
        for i in range(self.q):
            for m in range(self.m):
                alpha_val = self.alpha_im0(self.t_max, 0, i, m, params)
                # Условие на вещественную и мнимую части
                residuals.append(alpha_val.real)
                residuals.append(alpha_val.imag)

        # Условие 2: χ_12^0(tf, t0) = φ
        chi_val = self.chi_ij0(self.t_max, 0, 0, 1, params)
        residuals.append(chi_val - target_phi)
        self.iterations += 1
        print(f"Номер итерации {self.iterations}, норма невязки: {np.linalg.norm(np.array(residuals))}")

        return np.array(residuals)

    def solve(self, target_phi=np.pi / 2, initial_guess=None):
        """Решение системы нелинейных уравнений"""

        # Проверка размерности
        n_equations = self.q * self.m * 2 + 1
        print(f"Число уравнений: {n_equations}, число параметров: {self.n_params}")

        if n_equations != self.n_params:
            print(f"Внимание: система {'недоопределена' if n_equations < self.n_params else 'переопределена'}")

        # Решение системы уравнений
        solution = root(self.equations, initial_guess, args=(target_phi,),
                        method='hybr', options={'xtol': 1e-6})

        if solution.success:
            print("Решение найдено успешно!")
            self.solution_params = solution.x
            return solution.x
        else:
            print("Решение не найдено:", solution.message)
            return None

    def verify_solution(self, params, target_phi=np.pi / 2):
        """Проверка выполнения условий для найденного решения"""
        print("\nПроверка решения:")

        # Проверка условий на α
        for i in range(self.q):
            for m in range(self.m):
                alpha_val = self.alpha_im0(self.t_max, 0, i, m, params)
                print(f"α_{i}{m}^0(tf, t0) = {alpha_val:.6f} (должен быть 0)")

        # Проверка условия на χ
        chi_val = self.chi_ij0(self.t_max, 0, 0, 1, params)
        print(f"χ_01^0(tf, t0) = {chi_val:.6f} (целевое значение: {target_phi:.6f})")
        print(f"Ошибка: {abs(chi_val - target_phi):.6f}")

    def plot_pulse(self, params):
        """Построение графика найденного импульса Ω(t)"""
        t_values = np.linspace(0, self.t_max, 1000)
        omega_values = [shape_param.calc(t, params) for t in t_values]

        plt.figure(figsize=(10, 6))
        plt.plot(t_values, omega_values, 'b-', linewidth=2)
        plt.xlabel('Время')
        plt.ylabel('Ω(t)')
        plt.title('Форма импульса для реализации MS-гейта')
        plt.grid(True)
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Создаем решатель
    solver = GatePulseSolver(t_max=c.t_max, n_params=c.n_params, m=c.m, q=c.q, eta=c.eta,
                             omega_t=c.omega_t, mu=c.mu, psi=c.psi)

    solver.verify_solution([2 * np.pi / c.eta[0, 0] / c.t_max, 0, 0, 0, 0], np.pi / 2)
