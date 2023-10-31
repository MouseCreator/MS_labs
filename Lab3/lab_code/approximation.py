import sympy as sp
import numpy as np

from lab_code.io_reader import read


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
    vect = a_matrix * sp.Matrix(y_vector)
    b = sp.Matrix(beta)
    b_matrix = vect.jacobian(b)
    return b_matrix


def put_vals(matrix, y_vector, y_values, row_index):
    updated_matrix = matrix.copy()

    for i, symbol in enumerate(y_vector):
        updated_matrix = updated_matrix.subs(symbol, y_values[i, row_index])

    return updated_matrix


def calculate_u(a_matrix, b_matrix, y_values, y_symbols):
    n = y_values.shape[0]
    u_init = np.zeros(b_matrix.shape)
    u_curr = u_init.copy()
    delta = 0.2
    y_approx = to_vector(y_values, 0)
    u_map = {}
    for i in range(n):
        b_matrix_sub = np.array(
            b_matrix.subs({y_sym: y_value for y_sym, y_value in zip(y_symbols, y_approx)}))
        k1 = np.dot(a_matrix, u_curr) + b_matrix_sub
        k2 = np.dot(a_matrix, (u_curr + 0.5 * k1)) + b_matrix_sub
        k3 = np.dot(a_matrix, (u_curr + 0.5 * k2)) + b_matrix_sub
        k4 = np.dot(a_matrix, u_curr + k3) + b_matrix_sub
        u_curr = u_curr + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y_approx = move_y(a_matrix, y_approx)
        u_map[i] = u_curr.copy()
    return u_map


def create_approximation(a_matrix, y_matrix):
    n = a_matrix.shape[1]
    y_init = to_vector(y_matrix, 0)
    y_current = y_init.copy()
    delta = 0.2
    for i in range(n):
        k1 = np.dot(a_matrix, y_current)
        k2 = np.dot(a_matrix, y_current + 0.5 * k1)
        k3 = np.dot(a_matrix, y_current + 0.5 * k2)
        k4 = np.dot(a_matrix, y_current + k3)
        y_current = y_current + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_current


def to_vector(matrix, row):
    if 0 <= row < len(matrix):
        return matrix[row]
    else:
        print(f"Row index {row} is out of bounds.")
        return None


def integrate_sq_deviation(y_matrix, a_filled):
    delta_t = 0.2
    n = y_matrix.shape[0]
    integral_sum = 0
    y_approx = to_vector(y_matrix, 0)
    for i in range(n):
        vector_difference = to_vector(y_matrix, i) - y_approx
        partial = np.dot(vector_difference, vector_difference)
        integral_sum += partial
        y_approx = move_y(a_filled, y_approx)
    return integral_sum * delta_t


def integrate_delta_b(y_matrix, a_full, u_matrix):
    n = y_matrix.shape[0]
    u_sq = np.zeros((3, 3))
    vect = np.zeros(3)
    y_approx = to_vector(y_matrix, 0)
    for i in range(n):
        u_sub = u_matrix[i]

        partial = np.dot(u_sub.T, u_sub)
        u_sq = u_sq + partial

        exp = to_vector(y_matrix, i)
        vector_difference = exp - y_approx

        r = np.dot(u_sub.T, vector_difference)
        vect = vect + r
        y_approx = move_y(a_full, y_approx)

    print("vect:", vect)
    print("u_sq:")
    print(u_sq)
    u_sq = u_sq.astype('float64')
    u_sq_inv = np.linalg.inv(u_sq)
    result = np.dot(u_sq_inv, vect)

    return result


def update_beta(beta_init, delta_b):
    c1, c2, c3 = sp.symbols('c1 c2 c3')
    beta_init[c1] += delta_b[0]
    beta_init[c2] += delta_b[1]
    beta_init[c3] += delta_b[2]
    return beta_init


def move_y(a_full, y_approx, delta_t=0.2):
    k1 = np.dot(a_full, y_approx)
    k2 = np.dot(a_full, (y_approx + k1 / 2))
    k3 = np.dot(a_full, (y_approx + k2 / 2))
    k4 = np.dot(a_full, (y_approx + k3))
    return y_approx + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def move_u(u_current, a_full, b_current, t=0.2):
    k1 = np.dot(a_full, u_current) + b_current
    k2 = np.dot(a_full, (u_current + k1 / 2)) + b_current
    k3 = np.dot(a_full, (u_current + k2 / 2)) + b_current
    k4 = np.dot(a_full, (u_current + k3)) + b_current
    return u_current + t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def single_calculator(y_matrix, params, beta, beta_init, eps):
    a_matrix = create_matrix_and_symbols()
    a_filled = a_matrix.subs(params)

    b_len = len(beta)

    n = y_matrix.shape[0]
    m = y_matrix.shape[1]

    delta = 0.2

    while True:
        to_inverse = np.zeros((b_len, b_len))
        to_multiply = np.zeros(b_len)
        y_curr = to_vector(y_matrix, 0)
        u_current = np.zeros((m, b_len))
        a_current = np.array(a_filled.subs(beta_init)).astype('float64')

        for i in range(n):
            vect = sp.Matrix(a_filled) * sp.Matrix(y_curr)
            b = sp.Matrix(beta)
            b_current = np.array(vect.jacobian(b)).astype('float64')
            u_current = move_u(u_current, a_current, b_current)
            to_inverse = to_inverse + np.dot(u_current.T, u_current)
            to_multiply = to_multiply + np.dot(u_current.T, to_vector(y_matrix, i) - y_curr)

            y_curr = move_y(a_current, y_curr)

        to_inverse *= delta
        to_multiply *= delta
        delta_beta = np.dot(np.linalg.inv(to_inverse), to_multiply)
        beta_init = update_beta(beta_init, delta_beta)
        print("Beta", beta_init)
        if np.linalg.norm(delta_beta) < eps:
            return beta_init


def calculate(y_matrix, params, beta, beta_init, eps):
    a_matrix = create_matrix_and_symbols()
    a_filled = a_matrix.subs(params)
    y_vector = sp.symbols('y1 y2 y3 y4 y5 y6')
    i = 0
    b_matrix = calculate_b_matrix(a_filled, beta, y_vector)
    while True:
        a_temp = np.array(a_filled.subs(beta_init))
        u_map = calculate_u(a_temp, b_matrix, y_matrix, y_vector)
        delta_b = integrate_delta_b(y_matrix, a_temp, u_map)
        beta_init = update_beta(beta_init, delta_b)
        i += 1
        if i > 10:  # integrate_sq_deviation(y_matrix, a_filled) < eps:
            return beta_init


mtx_input = read('../input\\y10.txt')

c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
approximated = single_calculator(mtx_input, {m1: 12, m2: 28, m3: 18, c4: 0.12},
                                 [c1, c2, c3], {c1: 0.1, c2: 0.1, c3: 0.4}, 1e-6)
print(approximated)
