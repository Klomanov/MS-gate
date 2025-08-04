import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Параметры системы
omega_0 = 1.0 * 2 * np.pi  # Частота перехода между уровнями иона
nu = 0.1 * 2 * np.pi  # Частота колебательной моды (ЦМ)
Omega = 0.05 * 2 * np.pi  # Раби-частота лазера
eta = 0.1  # Параметр Ламба-Дике
delta = omega_0 - nu  # Расстройка (лазер настроен на красную боковую полосу)

# Время моделирования
t_max = 100
times = np.linspace(0, t_max, 500)

# Операторы
# Для двух ионов: спины и колебательная мода
N = 10  # Число фоковских состояний для моды

# Операторы Паули для каждого иона
sigma_z1 = tensor(sigmaz(), qeye(2), qeye(N))
sigma_z2 = tensor(qeye(2), sigmaz(), qeye(N))
sigma_p1 = tensor(sigmap(), qeye(2), qeye(N))
sigma_p2 = tensor(qeye(2), sigmap(), qeye(N))
sigma_m1 = tensor(sigmam(), qeye(2), qeye(N))
sigma_m2 = tensor(qeye(2), sigmam(), qeye(N))

# Операторы колебательной моды
a = tensor(qeye(2), qeye(2), destroy(N))
a_dag = tensor(qeye(2), qeye(2), create(N))


# Гамильтониан
def H(t):
    H0 = (0.5 * omega_0 * (sigma_z1 + sigma_z2) +
          nu * (a_dag * a + 0.5 * tensor(qzero(2), qzero(2), qeye(N))))
    return H0 + 0.5 * Omega * (
            sigma_p1 * (1 + 1j * eta * (a + a.dag())) * np.exp(-1j * delta * t) +
            sigma_p2 * (1 + 1j * eta * (a + a.dag())) * np.exp(-1j * delta * t) +
            # Эрмитово сопряжение (h.c.) для первого и второго ионов:
            sigma_m1 * (1 - 1j * eta * (a + a.dag())) * np.exp(1j * delta * t) +
            sigma_m2 * (1 - 1j * eta * (a + a.dag())) * np.exp(1j * delta * t)
    )


# Начальное состояние: оба иона в основном состоянии, 0 фононов
psi0 = tensor(basis(2, 0), basis(2, 0), basis(N, 0))

# Решение уравнения Шредингера
result = sesolve(H, psi0, times, options=qutip.Options(store_states=True))

# Вычисление населенностей
pop1 = expect(tensor(basis(2, 1).proj(), qeye(2), qeye(N)), result.states)
pop2 = expect(tensor(qeye(2), basis(2, 1).proj(), qeye(N)), result.states)
phonons = expect(a_dag * a, result.states)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(times, pop1, label="$|1 \\rangle \ Ион \ 1$")
plt.plot(times, pop2, label="$|1 \\rangle \ Ион \ 2$")
plt.plot(times, phonons, label="Число фононов")
plt.xlabel("Время")
plt.ylabel("Населенность")
plt.legend()
plt.grid()
plt.show()
