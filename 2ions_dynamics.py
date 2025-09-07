import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Параметры системы
omega_0 = 40 * 2 * np.pi  # Частота перехода между уровнями иона
omega_t = 0.1 * 2 * np.pi  # Частота колебательной моды (ЦМ)
eta = 0.1  # Параметр Ламба-Дике
delta_t = 0.01 * 2 * np.pi # Отстройка от боковых полос
omega_l1 = omega_0 + omega_t + delta_t  # Частота лазера 1
omega_l2 = omega_0 - omega_t - delta_t  # Частота лазера 2


# Время моделирования
t_max = 4*np.pi/delta_t
times = np.linspace(0, t_max, 1500)

Omega = 2*np.pi/eta/t_max  # Раби-частота

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
sigma_y1 = tensor(sigmay(), qeye(2), qeye(N))
sigma_y2 = tensor(qeye(2), sigmay(), qeye(N))

sigma_x1 = tensor(sigmax(), qeye(2), qeye(N))
sigma_x2 = tensor(qeye(2), sigmax(), qeye(N))

# Операторы колебательной моды
a = tensor(qeye(2), qeye(2), destroy(N))
a_dag = tensor(qeye(2), qeye(2), create(N))


# Гамильтониан противонаправленных пучков в представлении взаимодействия
def H(t):
    # H0 = (0.5 * omega_0 * (sigma_z1 + sigma_z2) +
    #       omega_t * (a_dag * a + 0.5 * tensor(qeye(2), qeye(2), qeye(N))))
    return 0.5*eta*Omega*(sigma_y1 + sigma_y2) * (a*np.exp(1j*delta_t*t) + a.dag()*np.exp(-1j*delta_t*t))


# Начальное состояние: оба иона в основном состоянии, 0 фононов
psi0 = tensor(basis(2, 0), basis(2, 0), basis(N, 0))

# Решение уравнения Шредингера
result = sesolve(H, psi0, times, options=qutip.Options(store_states=True))

# Вычисление населенностей
# pop11 = expect(tensor(basis(2, 1).proj(), basis(2, 1).proj(),  basis(N, 0).proj()), result.states)
# pop01 = expect(tensor(basis(2, 0).proj(), basis(2, 1).proj(),  basis(N, 0).proj()), result.states)
# pop10 = expect(tensor(basis(2, 1).proj(), basis(2, 0).proj(),  basis(N, 0).proj()), result.states)
# pop00 = expect(tensor(basis(2, 0).proj(), basis(2, 0).proj(),  basis(N, 0).proj()), result.states)

state11 = tensor(basis(2, 1), basis(2, 1), basis(N, 0))
state00 = tensor(basis(2, 0), basis(2, 0), basis(N, 0))

pop11, pop00 = [], []

for i in result.states:
    pop11.append(np.abs(i.overlap(state11))**2)
    pop00.append(np.abs(i.overlap(state00))**2)

phonons = expect(a_dag * a, result.states)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(times, pop11, label="11")
# plt.plot(times, pop01, label="01")
# plt.plot(times, pop10, label="10")
plt.plot(times, pop00, label="00")
plt.plot(times, phonons, label="Число фононов")
plt.xlabel("Время")
plt.ylabel("Населенность")
plt.legend()
plt.grid()
plt.show()
