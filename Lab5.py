import pandas as pd
import sympy as sp
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def drawFunc(lambda_function, start=-5, end=5):
    # 1. Точки для графика
    x_points = np.linspace(start, end, 500)
    y_points = lambda_function(x_points)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 2. Нарисовать функцию
    ax.plot(x_points, y_points, color='blue', label='f(x)')

    # 3. Оси через ноль
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 4. Отметки на X — каждое целое
    ax.set_xticks(np.arange(start, end + 1, 1))

    # 5. Отметки на Y — разумные, только несколько
    y_min = np.floor(min(y_points))
    y_max = np.ceil(max(y_points))
    ax.set_yticks(np.linspace(y_min, y_max, 6))  # 6 делений по Y

    # 6. Сетка только для основных отметок
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

    # 7. Легенда
    ax.legend()

    plt.show()

def drawFuncContour(f1, f2, startX=-5, endX=5, startY=-5, endY=5):
    x = np.linspace(startX, endX, 200)
    y = np.linspace(startY, endY, 200)
    X, Y = np.meshgrid(x, y)

    Z1 = f1(X, Y)
    Z2 = f2(X, Y)

    plt.figure(figsize=(7, 6))

    # Рисуем линии уровня = 0
    c1 = plt.contour(X, Y, Z1, levels=[0])
    c2 = plt.contour(X, Y, Z2, levels=[0])

    plt.clabel(c1)
    plt.clabel(c2)

    plt.axhline(0)
    plt.axvline(0)

    plt.grid(True)
    plt.title("Графическое решение системы")

    plt.show()
def bisection_method(f, a, b, eps=0.001):
    iterations = []
    n = 0
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        print("На интервале [a,b] нет корня или их четное число")
        return None, None

    while True:
        n += 1
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a)
        iterations.append([n, a, b, c, fc, error])

        # Условие остановки
        if abs(fc) < eps or error / 2 < eps:
            break

        # Выбор следующего интервала
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    # Создаем таблицу Pandas
    df = pd.DataFrame(iterations, columns=['iter', 'a', 'b', 'c', 'f(c)', 'error'])
    return c, df

def secant_method(f, x0, x1, eps=0.001, max_iter=100):
    iterations = []

    for n in range(1, max_iter + 1):
        f_x0 = f(x0)
        f_x1 = f(x1)

        # Проверка деления на 0
        if f_x1 - f_x0 == 0:
            print("Деление на ноль в методе хорд")
            return None, None

        # Формула хорд
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        f_x2 = f(x2)

        error = abs(x2 - x1)

        iterations.append([n, x0, x1, x2, f_x2, error])

        # Условие остановки
        if error < eps or abs(f_x2) < eps:
            break

        # Сдвиг
        x0, x1 = x1, x2

    df = pd.DataFrame(iterations, columns=['iter', 'x0', 'x1', 'x2', 'f(x2)', 'error'])
    return x2, df

def newton_method(f, f_prime, x0, eps=0.001, max_iter=100):
    iterations = []

    for n in range(1, max_iter + 1):
        f_x = f(x0)
        f_p = f_prime(x0)

        # Проверка деления на 0
        if f_p == 0:
            print("Производная равна 0 — метод Ньютона не работает")
            return None, None

        # Формула Ньютона
        x1 = x0 - f_x / f_p

        error = abs(x1 - x0)

        iterations.append([n, x0, f_x, f_p, x1, error])

        # Условие остановки
        if error < eps or abs(f_x) < eps:
            break

        x0 = x1

    df = pd.DataFrame(iterations, columns=['iter', 'x_n', 'f(x_n)', "f'(x_n)", 'x_next', 'error'])
    return x1, df

def simple_iteration_method(F, x0, eps=0.001, max_iter=100):
    iterations = []

    for n in range(1, max_iter + 1):
        x1 = F(x0)
        error = abs(x1 - x0)

        iterations.append([n, x0, x1, error])

        if error < eps:
            break

        x0 = x1

    df = pd.DataFrame(iterations, columns=['iter', 'x_n', 'x_next', 'error'])
    return x1, df

def task1():
    print('Задание 1')

    x = sp.Symbol('x')
    f_sym = 2.74 * x ** 3 - 1.93 * x ** 2 - 15.28 * x - 3.72
    f_num = sp.lambdify(x, f_sym, modules='numpy')
    print(sp.latex(f_sym))

    drawFunc(f_num)

    a = -3
    b = 4
    root, table = bisection_method(f_num, a, b)

    print('-' * 50)
    print(f"\nНайденный корень путем метода половинного деления: {root:.3f}")
    print(table)# нашли интервалы

    print('-' * 50, '\n\nКорни найденные методом хор на промежутках:')
    intervals = [(-2, -1), (-1, 0), (2, 3)]
    for a, b in intervals:
        root, table = secant_method(f_num, a, b)
        print(f"\nИнтервал [{a}, {b}]")
        print(f"Корень: {root:.3f}")
        print(table)

    print('-' * 50, '\n\nКорни найденные методом Ньютона на промежутках:')
    f_prime = sp.diff(f_sym, x)
    f_prime_num = sp.lambdify(x, f_prime, 'numpy')

    intervals = [(-2, -1), (-1, 0), (2, 3)]

    for a, b in intervals:
        x0 = (a + b) / 2  # середина интервала

        root, table = newton_method(f_num, f_prime_num, x0)

        print(f"\nИнтервал [{a}, {b}]")
        print(f"Начальное приближение: {x0}")
        print(f"Корень: {root:.3f}")
        print(table)

    print('-' * 50, '\nКорни найденный методом последовательных приближений на промежутках:')

    def F(x):
        return ((1.93 * x ** 2 + 15.28 * x + 3.72) / 2.74) ** (1 / 3)

    for a, b in intervals:
        x0 = (a + b) / 2  # старт как в Ньютоне

        root, table = simple_iteration_method(F, x0)

        print(f"\nИнтервал [{a}, {b}]")
        print(f"Начальное приближение: {x0}")
        print(f"Корень: {root:.3f}")
        print(table)

    print('-' * 50, '\nКорни, найденные через solve()', sp.solve(f_sym))



def task2():
    print('=' * 50, '\n\nЗадание 2')

    # Символы
    x, y = sp.symbols('x y')

    # Символьные функции
    f1_sym = sp.sin(x + 1) - y - 1.2
    f2_sym = 2 * x + sp.cos(y) - 2

    # Числовые функции
    f1_num = sp.lambdify((x, y), f1_sym, 'numpy')
    f2_num = sp.lambdify((x, y), f2_sym, 'numpy')

    drawFuncContour(f1_num, f2_num)

    # Матрица Якоби
    F = sp.Matrix([f1_sym, f2_sym])
    vars = sp.Matrix([x, y])
    J = F.jacobian(vars)
    print("Матрица Якоби:")
    sp.pprint(J)

    # Начальное приближение (примерно с графика)
    x0 = 0.5
    y0 = -0.3

    eps = 0.001
    max_iter = 50
    iteration = 0

    print("\nИтерации метода Ньютона:")
    print(f"{'iter':>4} {'x':>10} {'y':>10} {'dx':>10} {'dy':>10} {'||delta||':>12}")

    while iteration < max_iter:
        iteration += 1

        # Значения функций в текущей точке
        F_val = np.array([f1_num(x0, y0), f2_num(x0, y0)], dtype=float)

        # Значения Якоби в текущей точке
        J_val = np.array(J.subs({x: x0, y: y0}), dtype=float)

        # Решаем систему методом Крамера
        D = np.linalg.det(J_val)
        if abs(D) < 1e-12:
            print("Определитель близок к нулю, метод не применим")
            break

        # Определители для Крамера
        J_dx = J_val.copy()
        J_dx[:, 0] = -F_val
        dx = np.linalg.det(J_dx) / D

        J_dy = J_val.copy()
        J_dy[:, 1] = -F_val
        dy = np.linalg.det(J_dy) / D

        # Обновляем переменные
        x1 = x0 + dx
        y1 = y0 + dy

        delta_norm = np.sqrt(dx ** 2 + dy ** 2)

        print(f"{iteration:>4} {x1:10.5f} {y1:10.5f} {dx:10.5f} {dy:10.5f} {delta_norm:12.5f}")

        if delta_norm < eps:
            break

        x0, y0 = x1, y1

    print(f"\nРешение системы:\nx = {x1:.5f}, y = {y1:.5f}, найдено за {iteration} итераций")

    # Проверка решения подстановкой
    f1_check = f1_num(x1, y1)
    f2_check = f2_num(x1, y1)
    print(f"\nПроверка: f1(x,y) = {f1_check:.5f}, f2(x,y) = {f2_check:.5f}")

    # Для точного сравнения можно использовать sympy.nsolve
    sol = sp.nsolve([f1_sym, f2_sym], (x, y), (0.5, -0.3))
    print(f"\nsympy.nsolve решение: x = {sol[0]:.5f}, y = {sol[1]:.5f}")

def stepCheck(M, a, b, h):
    return M * h**2 * abs(b - a) / 12

def integrationStepCalc(a, b, f_sym, x, eps=0.001):
    # 1. Вторая производная
    f2 = sp.diff(f_sym, x, 2)
    f2_num = sp.lambdify(x, f2, 'numpy')

    # 2. Оценка максимума
    xs = np.linspace(a, b, 1000)
    M = np.max(np.abs(f2_num(xs)))

    tick_count = 4

    while True:
        h = abs(b - a) / tick_count

        if stepCheck(M, a, b, h) < eps:
            return h, tick_count, M

        tick_count += 4

def trapezoidalIntegration(a, b, f_num, h, n):
    x = np.linspace(a, b, n + 1)
    y = f_num(x)

    y[0] /= 2
    y[-1] /= 2

    return h * np.sum(y)

def simpsonIntegration(a, b, f_num, h, n):
    x = np.linspace(a, b, n + 1)
    y = f_num(x)

    # коэффициенты
    y[1:-1:2] *= 4   # нечётные индексы
    y[2:-1:2] *= 2   # чётные индексы (кроме первого и последнего)

    return (h / 3) * np.sum(y)

def task3():
    print('=' * 50, '\n\nЗадание 3')

    x = sp.symbols('x')
    f_sym = x**3 / sp.sqrt(1 - x**2)
    f_num = sp.lambdify(x, f_sym, 'numpy')

    a = -0.5
    b = 0.5

    h, n, M = integrationStepCalc(a, b, f_sym, x)

    # --- Трапеции ---
    h_trap = trapezoidalIntegration(a, b, f_num, h, n)
    h2_trap = trapezoidalIntegration(a, b, f_num, h * 2, n // 2)
    delta_trap = np.abs(h_trap - h2_trap) / 3

    # --- Симпсон ---
    h_simp = simpsonIntegration(a, b, f_num, h, n)
    h2_simp = simpsonIntegration(a, b, f_num, h * 2, n // 2)
    delta_simp = np.abs(h_simp - h2_simp) / 15

    print('h =', h)
    print('n =', n)

    print('-' * 50, '\n\nТрапеции')
    print('I(h) =', h_trap)
    print('I(2h) =', h2_trap)
    print('Погрешность =', delta_trap)

    print('-' * 50, '\n\nСимпсон')
    print('I(h) =', h_simp)
    print('I(2h) =', h2_simp)
    print('Погрешность =', delta_simp)

    print('-' * 50, '\n\nПроверка через функцию')
    result, error = quad(f_num, a, b)
    print(f'I = {result:.6f}')
    print(f'Оценка ошибки = {error:.2e}')

    print('-' * 50, '\n\nПроверка через подсчет по формуле Ньютона-Лейбница')
    print('При интегрировании получаем следующую формулу:')
    sp.pprint(sp.integrate(f_sym, x))
    print('Соответственно в результате при подстановке получаем 0')

def f(x, y):
    return 0.5 * (x - 1) * np.exp(x) * y**2 - x * y

def runge_kutta_4(f, a, b, y0, h):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)

    y[0] = y0

    for k in range(n):
        F1 = f(x[k], y[k])
        F2 = f(x[k] + h/2, y[k] + h/2 * F1)
        F3 = f(x[k] + h/2, y[k] + h/2 * F2)
        F4 = f(x[k] + h, y[k] + h * F3)

        y[k+1] = y[k] + h/6 * (F1 + 2*F2 + 2*F3 + F4)

    return x, y

def euler_method(f, a, b, y0, h):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)

    y[0] = y0

    for k in range(n):
        y[k+1] = y[k] + h * f(x[k], y[k])

    return x, y

def find_step(f, a, b, y0, eps=1e-4):
    h = 0.5  # начальный шаг

    while True:
        x1, y1 = runge_kutta_4(f, a, b, y0, h)
        x2, y2 = runge_kutta_4(f, a, b, y0, h / 2)

        # сравниваем значения в узлах
        diff = np.max(np.abs(y1 - y2[::2]))

        if diff < eps:
            return h
        h /= 2

def exact_solution(x):
    return 4 * np.exp(-x**2 / 2) / (1 + np.exp(x))

def task4():
    print('=' * 50, '\n\nЗадание 4')

    a = 0
    b = 2
    y0 = 2

    # шаг
    h = find_step(f, a, b, y0)
    print('Найденный шаг h =', h)

    # решения
    x_rk, y_rk = runge_kutta_4(f, a, b, y0, h)
    x_eu, y_eu = euler_method(f, a, b, y0, h)

    # точное
    y_exact = exact_solution(x_rk)

    # ошибки
    err_rk = np.abs(y_rk - y_exact)
    err_eu = np.abs(y_eu - y_exact)

    print('Максимальная ошибка RK4 =', np.max(err_rk))
    print('Максимальная ошибка Euler =', np.max(err_eu))

    # --- Таблица ---
    df = pd.DataFrame({
        'x': x_rk,
        'RK4': y_rk,
        'Euler': y_eu,
        'Exact': y_exact,
        'Err RK4': err_rk,
        'Err Euler': err_eu
    })

    print('\nТаблица:')
    print(df.head(10))  # можно убрать .head()

    # --- График ---
    plt.figure()
    plt.plot(x_rk, y_rk, label='Runge-Kutta 4')
    plt.plot(x_eu, y_eu, label='Euler')
    plt.plot(x_rk, y_exact, '--', label='Exact')

    plt.legend()
    plt.grid()
    plt.title('Решение задачи Коши')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # --- scipy ---
    sol = solve_ivp(f, [a, b], [y0], t_eval=x_rk)

    print('-' * 50, '\n\nПроверка через solve_ivp')
    print('Макс отклонение от RK4 =',
          np.max(np.abs(sol.y[0] - y_rk)))

if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()