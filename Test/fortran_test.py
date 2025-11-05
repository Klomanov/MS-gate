import add
import numpy as np

# Создаём массивы комплексных чисел
a = np.array([1+0j, 3+0j])
b = np.array([5+0j, 6+0j])
c = np.zeros_like(a)  # массив для результата

# Вызываем Fortran-функцию
add.zadd([1, 2, 3], [1, 2], [3, 4], 3)

print(c)