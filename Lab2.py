import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

rng = np.random.default_rng()

def task1():
    print('Task 1')
    matrix = rng.integers(low=-3, high=3, size=(10,10), endpoint=True)
    print(matrix)

    sub_matrix = matrix[1:5,6:10]
    print(sub_matrix)
    print(np.linalg.det(sub_matrix))

def task2():
    print('=' * 50, '\nTask 2')

    matrix1 = rng.uniform(2, 3, size=(3, 3))
    matrix2 = rng.uniform(2, 3, size=(3, 3))
    print('Matrix 1\n', matrix1)
    print('Matrix 2\n', matrix2)

    # Способ 1: Векторный алгоритм умножения матриц
    # (каждый элемент вычисляется как скалярное произведение строки и столбца)
    print('\n' + '=' * 50)
    print('1. Векторный алгоритм умножения матриц:')

    result_vector = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Скалярное произведение i-й строки matrix1 и j-го столбца matrix2
            result_vector[i, j] = np.sum(matrix1[i, :] * matrix2[:, j])

    print('Результат:\n', result_vector)

    # Способ 2: Матричный алгоритм с записью по столбцам
    print('\n' + '=' * 50)
    print('2. Матричный алгоритм (запись по столбцам):')

    result_column = np.zeros((3, 3))
    for j in range(3):  # Идем по столбцам результирующей матрицы
        for i in range(3):  # Идем по строкам
            result_column[i, j] = np.sum(matrix1[i, :] * matrix2[:, j])

    print('Результат:\n', result_column)

    # Способ 3: Проверка с помощью np.dot
    print('\n' + '=' * 50)
    print('3. Проверка с помощью np.dot:')

    result_dot = np.dot(matrix1, matrix2)
    print('Результат:\n', result_dot)

    # Проверка совпадения результатов
    print('\n' + '=' * 50)
    print('Проверка совпадения результатов:')

    print(f'Совпадение 1 и 2 способов: {np.allclose(result_vector, result_column)}')
    print(f'Совпадение 1 и 3 способов: {np.allclose(result_vector, result_dot)}')
    print(f'Совпадение 2 и 3 способов: {np.allclose(result_column, result_dot)}')

def task3():
    print('=' * 50, '\nTask 3')

    matrix = rng.integers(low=-10, high=10, size=(5, 5))

    # Обнуляем элементы ниже главной диагонали
    for i in range(5):
        for j in range(5):
            if i > j:  # элементы ниже главной диагонали
                matrix[i, j] = 0

    # Проверяем, что матрица невырожденная
    while np.linalg.det(matrix) == 0:
        print("Матрица вырожденная, перегенерируем...")
        matrix = rng.integers(low=-10, high=10, size=(5, 5))
        for i in range(5):
            for j in range(5):
                if i > j:
                    matrix[i, j] = 0

    # вектор B
    vector = rng.integers(low=-10, high=10, size=(5,))

    print('Верхнетреугольная матрица A:\n', matrix)
    print('Вектор B:\n', vector)

    print('\n' + '=' * 50)

    # Решаем систему AX = B
    result = np.linalg.solve(matrix, vector)
    print('Решение системы X:\n', result)

def task4():
    print('=' * 50, '\nTask 4')

    A = np.array([
        [4.4, -2.5, 19.2, -10.8],
        [5.5, -9.3, -14.2, 13.2],
        [7.1, -11.5, 5.3, -6.7],
        [14.2, 23.4, -8.8, 5.3]
    ])

    b = np.array([4.3, 6.8, -1.8, 7.2])

    try:
        # lu_factor возвращает компактное представление L и U, а также данные о перестановках (pivoting)
        lu, piv = sc.linalg.lu_factor(A)

        x = sc.linalg.lu_solve((lu, piv), b)

        # Вывод результата
        print("Результаты решения системы:")
        for i, val in enumerate(x, 1):
            print(f"x{i} = {val:.4f}")

        # Проверка (A * x должно быть равно b)
        check = np.dot(A, x)
        print("\nПроверка (A * x):", np.round(check, 2))

    except Exception as e:
        print(f"Ошибка при решении: {e}")

task1()
task2()
task3()
task4()
