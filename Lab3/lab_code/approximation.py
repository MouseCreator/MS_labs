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


def calculate_b_matrix(a_matrix, beta, y_vector):
    vect = a_matrix * y_vector
    b_matrix = vect.jacobian(beta)
    return b_matrix


def put_vals(matrix, y_vector, y_values, row_index):
    updated_matrix = matrix.copy()

    for i, symbol in enumerate(y_vector):
        updated_matrix = updated_matrix.subs(symbol, y_values[i, row_index])

    return updated_matrix


def calculate_u(a_matrix, b_matrix, y_vector, y_values):
    n = a_matrix.shape[1]
    u_init = np.zeros(b_matrix.shape)
    u_curr = u_init.copy()
    t = sp.symbols('t')
    time = 0
    delta = 0.2
    for i in range(n):
        b_matrix_sub = put_vals(b_matrix, y_vector, y_values, i)
        k1 = (a_matrix * u_curr) + b_matrix_sub
        k1_t = np.array(k1.subs({t: time}))
        k2 = (a_matrix * (u_curr + 0.5 * k1)) + b_matrix_sub
        k2_t = np.array(k2.subs({t: time + delta / 2}))
        k3 = (a_matrix * (u_curr + 0.5 * k2)) + b_matrix_sub
        k3_t = np.array(k3.subs({t: time + delta / 2}))
        k4 = (a_matrix * u_curr + k3) + b_matrix_sub
        k4_t = np.array(k4.subs({t: time + delta}))
        time += delta
        u_curr = u_curr + delta / 6 * (k1_t + k2_t + k3_t + k4_t)
    return u_curr


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


def integrate_delta_b(y_matrix, y_approximation, u_matrix):
    delta_t = 0.2
    n = y_matrix.shape[1]
    u_sq = 0
    t = sp.symbols('t')
    time = 0
    vect = np.zeros(n)
    for i in range(n):
        u_sub = np.array(u_matrix.subs(t, time))
        partial = np.dot(u_sub.T, u_sub)
        u_sq += partial

        vector_difference = to_vector(y_matrix, i) - to_vector(y_approximation, i)
        vect += np.dot(u_sub.T, vector_difference)

        time += delta_t
    u_sq_inv = np.linalg.inv(u_sq)
    return np.dot(u_sq_inv, vect)


def calculate(y_matrix, params, beta, beta_init, eps):
    y_vector = sp.symbols('y1 y2 y3 y4 y5 y6')
    a_matrix = create_matrix_and_symbols()
    a_matrix = a_matrix.subs(params)
    y_approximation = a_matrix.subs(beta_init)
    b_matrix = calculate_b_matrix(a_matrix, beta, y_vector)
    while True:
        u_matrix = calculate_u(a_matrix, b_matrix, y_vector, y_matrix)
        delta_b = integrate_delta_b(y_matrix, y_approximation, u_matrix)
        if integrate_sq_deviation(y_matrix, y_approximation) < eps:
            return y_approximation


