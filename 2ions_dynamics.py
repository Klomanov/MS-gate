import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Параметры системы
m = 2  # Количество атомов в цепи
omega_0 = 100 * 2 * np.pi  # Частота перехода между уровнями иона
omega_t = [10 * 2 * np.pi, 15 * 2 * np.pi]  # Частота колебательной моды (ЦМ)
eta = [0.1, 0.2]  # Параметр Ламба-Дике
delta_t = 0.1 * 2 * np.pi # Отстройка от боковых полос
omega_l1 = omega_0 + omega_t + delta_t  # Частота лазера 1
omega_l2 = omega_0 - omega_t - delta_t  # Частота лазера 2


assert m == len(omega_t) and m == len(eta)

# Время моделирования
t_max = 4*np.pi/delta_t
times = np.linspace(0, t_max, 1500)

Omega = 2*np.pi/eta/t_max  # Раби-частота

# Операторы
# Для двух ионов: спины и колебательная мода
N = 10  # Число фоковских состояний для моды

# Операторы Паули для каждого иона
sigma_z1 = tensor(sigmaz(), qeye(2), qeye(m*N))
sigma_z2 = tensor(qeye(2), sigmaz(), qeye(m*N))
sigma_p1 = tensor(sigmap(), qeye(2), qeye(m*N))
sigma_p2 = tensor(qeye(2), sigmap(), qeye(m*N))
sigma_m1 = tensor(sigmam(), qeye(2), qeye(m*N))
sigma_m2 = tensor(qeye(2), sigmam(), qeye(m*N))
sigma_y1 = tensor(sigmay(), qeye(2), qeye(m*N))
sigma_y2 = tensor(qeye(2), sigmay(), qeye(m*N))

sigma_x1 = tensor(sigmax(), qeye(2), qeye(m*N))
sigma_x2 = tensor(qeye(2), sigmax(), qeye(m*N))

# Операторы колебательной моды
a = [tensor(qeye(2), qeye(2), destroy(N), qeye(N)), tensor(qeye(2), qeye(2), qeye(N), destroy(N))]


def H(t):
    # H0 = (0.5 * omega_0 * (sigma_z1 + sigma_z2) +
    #       omega_t * (a_dag * a + 0.5 * tensor(qeye(2), qeye(2), qeye(N))))

    # Counter-propagating lasers (Int picture)
    # return 0.5*Omega*(np.exp(1j*(omega_t+delta_t)*t) * (1 + 1j*eta*(
    # a*np.exp(-1j*omega_t*t) + a.dag()*np.exp(1j*omega_t*t)))+ np.exp(-1j*(omega_t+delta_t)*t) * (1 - 1j*eta*(
    # a*np.exp(-1j*omega_t*t) + a.dag()*np.exp(1j*omega_t*t))))*(sigma_x1+sigma_x2)

    # Co-propagating lasers (Int picture)
    # return Omega*np.cos((omega_t+delta_t)*t)*((sigma_p1+sigma_p2)*(1-1j*eta*(
    # a*np.exp(-1j*omega_t*t) + a.dag()*np.exp(1j*omega_t*t))) + (sigma_m1+sigma_m2)*(1+1j*eta*(a*np.exp(
    # -1j*omega_t*t) + a.dag()*np.exp(1j*omega_t*t))))

    H0 = Omega*np.cos((omega_t+delta_t)*t)*(sigma_x1 + sigma_x2)
    H1 = eta*Omega*np.cos((omega_t+delta_t)*t)*(a*np.exp(-1j*omega_t*t) + a.dag()*np.exp(1j*omega_t*t))*(sigma_y1 + sigma_y2)
    return H0 + H1

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

phonons = expect(a.dag() * a, result.states)


print("Fidelity:", fidelity(result.states[len(result.states)//2], (state00 + 1j*state11)/np.sqrt(2)))
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
