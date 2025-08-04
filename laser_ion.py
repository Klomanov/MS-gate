import numpy as np
import qutip
from qutip import sigmax, sigmaz, basis, sesolve

# Параметры
Delta = 1.0  # Расстройка
A0 = 2.0     # Амплитуда импульса
t0 = 5.0     # Центр импульса
sigma = 1.0  # Ширина импульса

# Временная зависимость
def H1_coeff(t, args):
    return A0 * np.exp(-(t - t0)**2 / (2 * sigma**2))

# Стационарная часть гамильтониана
H0 = (Delta / 2) * sigmaz()

# Временно-зависимая часть
H1 = sigmax()

# Полный гамильтониан (H0 + H1(t))
H = [H0, [H1, H1_coeff]]

# Начальное состояние (например, основное состояние)
psi0 = basis(2, 0)

# Временные точки для расчета
tlist = np.linspace(0, 10, 100)

# Решение уравнения Шрёдингера
result = sesolve(H, psi0, tlist, e_ops=[sigmaz()], options=qutip.Options(store_states=True))

p0 = []
p1 = []

for i in range(len(tlist)):
    psi_t = result.states[i]
    p0.append(np.abs(psi_t.full()[0][0])**2)
    p1.append(np.abs(psi_t.full()[1][0])**2)

# Визуализация
import matplotlib.pyplot as plt
plt.plot(tlist, p0)
plt.plot(tlist, p1)
plt.xlabel('Время')
plt.ylabel('<σ_z>')
plt.title('Динамика двухуровневой системы под действием импульса')
plt.show()