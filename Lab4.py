import numpy as np
from sympy import symbols, prod
import matplotlib.pyplot as plt

coords_table1 = np.array([
    [0.43, 1.63597],
    [0.48, 1.73234],
    [0.55, 1.87686],
    [0.62, 2.03345],
    [0.70, 2.22846],
    [0.75, 2.35973]
    ])
x_ansv1 = np.array([0.702, 0.512, 0.645, 0.736, 0.608])

coords_table2 = np.array([
    [1.415, 0.888551],
    [1.420, 0.889599],
    [1.425, 0.890637],
    [1.430, 0.891667],
    [1.435, 0.892687],
    [1.440, 0.893698],
    [1.445, 0.894700],
    [1.450, 0.895693],
    [1.455, 0.896677],
    [1.460, 0.897653],
    [1.465, 0.898619]
])
x_ansv2 = np.array([1.4161, 1.4625, 1.4135, 1.470])

def print_lagrange_formula(x_points, y_points):
    x = symbols('x')
    n = len(x_points)
    formula = 0

    for i in range(n):
        # Формируем L_i(x)
        li = prod([(x - x_points[j]) / (x_points[i] - x_points[j])
                   for j in range(n) if j != i])
        formula += y_points[i] * li

    print(f"Многочлен Лагранжа:\nP(x) = {formula}")
    return formula

def lagrange_polynomial(x_points, y_points, x):
    """
    Вычисляет значение интерполяционного многочлена Лагранжа в точке x.

    x_points: список узлов x_i
    y_points: список значений y_i
    x: точка, в которой вычисляем значение
    """
    n = len(x_points)
    result = 0.0

    for i in range(n):
        # Вычисляем базисный полином L_i(x)
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * li

    return result


def newton_forward(x_points, y_points, x):
    """
    Первая интерполяционная формула Ньютона (для начала таблицы)
    x_points: узлы (должны быть равноотстоящими)
    y_points: значения
    x: точка для вычисления
    """
    n = len(x_points)
    h = x_points[1] - x_points[0]  # шаг

    # Проверка на равноотстоящие узлы
    for i in range(1, n):
        if abs((x_points[i] - x_points[i - 1]) - h) > 1e-10:
            raise ValueError("Узлы должны быть равноотстоящими")

    # Вычисляем t
    t = (x - x_points[0]) / h

    # Построение таблицы конечных разностей
    diff_table = [y_points.copy()]
    for order in range(1, n):
        diff = []
        for i in range(n - order):
            diff.append(diff_table[order - 1][i + 1] - diff_table[order - 1][i])
        diff_table.append(diff)

    # Интерполяция по формуле Ньютона
    result = diff_table[0][0]
    term = 1.0
    factorial = 1

    for k in range(1, n):
        term *= (t - (k - 1))
        factorial *= k
        result += term / factorial * diff_table[k][0]

    return result

def printFunc(x_points, y_points):
    plt.plot(x_points, y_points)
    plt.show()


def task1():
    print('Задание 1')
    x_points = coords_table1[:, 0]
    y_points = coords_table1[:, 1]

    print_lagrange_formula(x_points, y_points)

    # Вычисляем значения для новых точек
    new_y = []
    for x in x_ansv1:
        y = lagrange_polynomial(x_points, y_points, x)
        new_y.append(y)
        print('Значение функции в точке', x, '=', y)

    # Добавляем новые точки к существующим
    x_points = np.append(x_points, x_ansv1)
    y_points = np.append(y_points, new_y)

    # Сортируем по x
    sort_idx = np.argsort(x_points)
    x_points = x_points[sort_idx]
    y_points = y_points[sort_idx]

    printFunc(x_points, y_points)

def task2():
    print('=' * 50, '\nЗадание 2')

    x_points = coords_table2[:, 0]
    y_points = coords_table2[:, 1]

    for i in range(len(x_ansv2)):
        print('Значение функции в точке', x_ansv2[i], '=', newton_forward(x_points, y_points, x_ansv2[i]))

if __name__ == "__main__":
    task1()
    task2()