import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def drawFunc(lambda_function):
    x_points = np.linspace(-5, 5, 100)
    y_points = lambda_function(x_points)

    # 3. Добавление сетки
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.plot(x_points, y_points)
    plt.show()

def task1():
    x = sp.Symbol('x')
    f_sym = 2.74 * x ** 3 - 1.93 * x ** 2 - 15.28 * x - 3.72
    f_num = sp.lambdify(x, f_sym, modules='numpy')
    print(sp.latex(f_sym))

    drawFunc(f_num)

task1()