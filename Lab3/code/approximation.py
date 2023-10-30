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
    symbols = [c1, c2, c3, c4, m1, m2, m3]
    return matrix_a, symbols


def to_vector(matrix, row):
    if 0 <= row < len(matrix):
        return matrix[row]
    else:
        print(f"Row index {row} is out of bounds.")
        return None


def integrate_sq_deviation(y_matrix, y_approximation):
    delta_t = 0.2
    n = y_matrix.shape[1]
    integral_sum = 0
    for i in range(n):
        vector_difference = to_vector(y_matrix, i) - to_vector(y_approximation, i)
        partial = np.dot(vector_difference, vector_difference)
        integral_sum += partial
    return integral_sum * delta_t
