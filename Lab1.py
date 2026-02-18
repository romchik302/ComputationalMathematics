import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def task1():
    matrix = np.random.rand(5,5) * 10

    print("1.\tСлучайная [0;10) матрица:\n", matrix)
    print("Транспонированная матрица:\n", np.transpose(matrix))
    print("Определитель матрицы:\n", np.linalg.det(matrix))

def task2():
    matrix1 = np.round(np.random.rand(5,5) * 10).astype(int)
    print("\n", "-" * 40, "\n\n2.\tСлучайная [1;10) матрица(целочисленная):\n", matrix1)

    vector1 = np.round(np.random.rand(5,1) * 10).astype(int)
    print("Случайный [1;10) вектор\n", vector1)
    print("Перемножение матрицы и вектора:\n", matrix1 * vector1)

def task3():
    x, y = sp.symbols('x y')
    expr = (2 * x + 3 * y) ** 2 - (4 * x * y) / 3 * (x - y)

    print("\n", "-" * 40, "\n\n3.\tИсходное выражение:")
    sp.pprint(expr)

    print("Упрощенное выражение:")
    expr = sp.cancel(expr)
    sp.pprint(expr)

    print("Результат уравнения при х = 1.038, у = sqrt(7):")
    sp.pprint(expr.subs({x: 1.038, y: np.sqrt(7)}))

def task4():
    x, y = sp.symbols('x y')
    expr = (2 * x + 3 * y) ** 2 - (4 * x * y) / 3 * (x - y)
    expr = sp.cancel(expr)

    print("\n", "-" * 40, "\n\n4.\tЧастная производная по х:")
    sp.pprint(sp.diff(expr, x))
    print("Частная производная по y:")
    sp.pprint(sp.diff(expr, y))

def task5():
    # Представим уравнение в виде A * X = B

    A_data = [[1, 0, -1],
         [-1, -1, 3],
         [1, -2, -4]]

    B_data = [[1],
         [-3],
         [5]]

    x1, x2, x3 = sp.symbols('x1 x2 x3')
    solution_np = np.linalg.solve(np.array(A_data), np.array(B_data))
    equations = [
        sp.Eq(A_data[0][0] * x1 + A_data[0][1] * x2 + A_data[0][2] * x3, B_data[0][0]),
        sp.Eq(A_data[1][0] * x1 + A_data[1][1] * x2 + A_data[1][2] * x3, B_data[1][0]),
        sp.Eq(A_data[2][0] * x1 + A_data[2][1] * x2 + A_data[2][2] * x3, B_data[2][0])
    ]
    solution_sp = sp.solve(equations)

    print("\n", "-" * 40, "\n\n5.\tУравнение имеет вид:")
    for line in equations:
        sp.pprint(line)

    print("Решение через SymPy:")
    sp.pprint(solution_sp)

    print("Решение через NumPy:")
    for res in solution_np:
        print(res[0])

def task6():
    x = sp.symbols('x')
    expr = sp.root(x, 2) + sp.root(x, 3) ** 2
    integral = sp.Integral(expr, x)

    print("\n", "-" * 40, "\n\n6.\tИнтеграл:")
    sp.pprint(integral)
    print("Равен:")
    sp.pprint(integral.doit())

def task7():
    x,y = sp.symbols('x y')

    y1 = -1
    y2 = 1
    x1 = 2 * y
    x2 = y

    expr = (x - y) * sp.exp(y)
    double_integral = sp.Integral(expr, (x, x1, x2), (y, y1, y2))

    print("\n", "-" * 40, "\n\n7.\tДвойной интеграл:")
    sp.pprint(double_integral)
    print("Решение через SymPy:")
    sp.pprint(double_integral.doit())
    print("Значение выражения для SymPy: ", double_integral.doit().evalf())

    result, error = sc.integrate.dblquad(
        lambda x, y: (x - y) * np.exp(y),
        y1, y2,
        lambda y: 2 * y,
        lambda y: y
    )
    print("Значение выражения для SciPy: ", result )

def task8():
    x = np.arange(-10, 10, 0.1)
    y1 = 3 * np.sin(x)
    y2 = np.sqrt(x + 5)

    # Точки пересечения
    x_intersect = []
    for x0 in [-2, 0, 2]:  # начальные приближения
        x_root = fsolve(lambda x: 3 * np.sin(x) - np.sqrt(x + 5), x0)[0]
        # Проверяем, что корень в области определения и не дублируется
        if x_root >= -5 and not any(abs(x_root - xi) < 0.1 for xi in x_intersect):
            x_intersect.append(x_root)

    y_intersect = [3 * np.sin(x) for x in x_intersect]


    fig, ax = plt.subplots()
    ax.plot(x, y1, label='3·sin(x)', color='b')
    ax.plot(x, y2, label='sqrt(x+5)', color='g')
    plt.title("Графики")
    plt.xlabel("X", loc='right')  # подпись справа
    plt.ylabel("Y", loc='top', rotation = 90)  # подпись сверху

    # Форматирование сетки
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.scatter(x_intersect, y_intersect, color='orange', s=50, label='Пересечения')
    plt.legend(loc='upper right')

    plt.show()

task1()
task2()
task3()
task4()
task5()
task6()
task7()
task8()