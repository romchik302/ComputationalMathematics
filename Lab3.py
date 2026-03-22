import numpy as np
import sympy as sp
rng = np.random.default_rng(42)
vec = rng.integers(low=-10, high=10, size=10)
print('Случайный вектор для заданий 1 и 3:\n', vec)

def householder_obliterate(x, start_idx):
    y = x.copy()
    a = y[start_idx:]

    sigma = np.linalg.norm(a)

    if sigma == 0:
        return y  # уже обнулено

    v = a.copy()
    v[0] = v[0] + np.sign(a[0]) * sigma

    v = v / np.linalg.norm(v)

    y[start_idx:] = a - 2 * np.dot(v, a) * v

    H = np.eye(len(x))
    H[start_idx:, start_idx:] -= 2 * np.outer(v, v)
    return y, H


def qr_householder(A):
    """QR-разложение методом Хаусхолдера"""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for j in range(min(m, n)):
        x = R[j:, j]

        sigma = np.linalg.norm(x)
        if sigma == 0:
            continue

        v = x.copy()
        v[0] += np.sign(x[0]) * sigma
        v = v / np.linalg.norm(v)

        H_j = np.eye(m)
        H_j[j:, j:] -= 2 * np.outer(v, v)

        R = H_j @ R
        Q = Q @ H_j.T

    return Q, R


def solve_qr(A, b):
    """Решение системы Ax = b через QR-разложение"""
    m, n = A.shape

    # QR-разложение
    Q, R = qr_householder(A)

    # Преобразуем правую часть: Q^T * b
    Qt_b = Q.T @ b

    # Обратная подстановка для Rx = Q^T b
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Qt_b[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]

    return x, Q, R

def task1():
    print('Задание 1:')
    sum = 0
    for val in vec:
        sum += val * val

    sum = np.sqrt(sum)

    print('Самостоятельно написанная функция: ', sum)
    print('Проверка через функцию np.linalg.norm(): ' ,np.linalg.norm(vec))

def task2():
    print('=' * 50, '\nЗадание 2')

    matrix = rng.integers(low=-10, high=10, size=(10,10))
    print('Случайная матрица:\n', matrix)

    trans_matrix = np.transpose(matrix)
    m_matrix = np.dot(trans_matrix, matrix)

    lambdas = np.array(np.linalg.eigvals(m_matrix))
    spec_norm = np.sqrt(np.max(lambdas))

    print('Значение полученное из собственного алгоритма: ', spec_norm)
    print('Значение через np.linalg.norm(): ', np.linalg.norm(matrix, ord = 2))

def task3():
    print('=' * 50, '\nЗадание 3')

    result, H = householder_obliterate(vec, 3)
    print("Исходный:   ", vec)
    print("Результат:  ", result)

def task4():
    print('=' * 50, '\nЗадание 4')

    matrix = np.array([[4.4, -2.5, 19.2, -10.8],
                       [5.5, -9.3, -14.2, 13.2],
                       [7.1, -11.5, 5.3, -6.7],
                       [14.2, 23.4, -8.8, 5.3]])

    b_vec = np.array([4.3, 6.8, -1.8, 7.2])
    x_vec, Q, R = solve_qr(matrix, b_vec)

    print('Q =', Q)
    print('R =', R)
    print('\nРешение:', x_vec)

    print('Проверка через numpy:')
    q_np, r_np = np.linalg.qr(matrix)
    np_res = np.linalg.solve(r_np, q_np.T @ b_vec)
    print(np_res)

task1()
task2()
task3()
task4()
