import numpy as np
from constants import Constants as c

"""
Параметризация импульса через синусы
"""


def basis_functions(t, k):
    """Базисные функции для параметризации Ω(t)"""
    # Используем многочлены для параметризации
    if k == 0:
        return 1
    return t**k
    # Используем синусы, чтобы автоматически удовлетворять Ω(t0)=Ω(tf)=0
    # return np.sin(np.pi * k * t / c.t_max)


def calc(t, params):
    """Ω(t) как линейная комбинация базисных функций"""
    result = 0.0
    for k in range(len(params)):
        result += params[k] * basis_functions(t, k)
    return result
