import sympy as sp
import numpy as np


def create_matrix_and_symbols():
    c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
    matrix_a = [
        [0, 1, 0, 0, 0, 0],
        [-(c2 + c1) / m1, 0, c2 / m1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [c2 / m2, 0, -(c2 + c3) / m3, 0, c3 / m2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, c3 / m3, 0, -(c4 + c3) / m3, 0]
    ]
    return sp.Matrix(matrix_a)


def print_matrix(matrix, name='Matrix'):
    print(f"\n{name}\n")
    num_rows, num_cols = matrix.shape
    max_widths = [max(len(str(matrix[i, j])) for i in range(num_rows)) for j in range(num_cols)]
    for i in range(num_rows):
        row_str = ""
        for j in range(num_cols):
            element = str(matrix[i, j])
            padding = max_widths[j] - len(element)
            row_str += element + ' ' * (padding + 1) + "\t\t"
        print(row_str)


def derivative_of_vector(y_vector, b_vector):
    derivatives = [sp.diff(Yi, Bi) for Yi in y_vector for Bi in b_vector]
    num_cols = len(b_vector)
    derivative_matrix = [derivatives[i:i + num_cols] for i in range(0, len(derivatives), num_cols)]
    return sp.Matrix(derivative_matrix)


def calculate_u():
    pass


def calculate(y_matrix, params, beta, beta_init, eps):
    a_matrix = create_matrix_and_symbols()
    a_filled = a_matrix.subs(params)
    print_matrix(a_filled)
    y1, y2, y3, y4, y5, y6 = sp.symbols('y1 y2 y3 y4 y5 y6')
    y_symbols = [y1, y2, y3, y4, y5, y6]
    ys = sp.Matrix(y_symbols).reshape(len(y_symbols), 1)
    ay_prod = a_filled * ys
    ay_db = derivative_of_vector(ay_prod, beta)
    print_matrix(ay_db, 'AY DB')
    return beta_init
