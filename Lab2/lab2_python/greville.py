import numpy as np


def calculate_first(a0):
    return a0 / np.dot(a0.T, a0)


def calculate_z(a_cur, a_inv):
    e = np.identity(a_cur.shape[0])
    return e - np.dot(a_inv, a_cur)


def multiply(*args):
    if len(args) == 0:
        return 0
    if len(args) == 1:
        return args[0]

    result = np.dot(args[0], args[1])
    for i in range(2, len(args)):
        result = np.dot(result, args[i])
    return result


def add_row(current_matrix, a):
    return np.append(current_matrix, [a], axis=0)


def append_to_inverse(inverse_matrix, a, z, denum):
    inverse_new = inverse_matrix - multiply(z, a, a.T, inverse_matrix) / denum
    row = multiply(z, a)
    return add_row(inverse_new, row)


def append_zero_case(inverse_matrix, a, z):
    r = multiply(inverse_matrix, inverse_matrix.T)
    denum = 1 + multiply(a.T, r, a)
    inverse_new = inverse_matrix - multiply(z, a, a.T, inverse_matrix) / denum
    row = multiply(r, a) / denum
    return add_row(inverse_new, row)


def greville(matrix):
    EPS = 0.000001
    a0_inv = calculate_first(matrix[0])
    n = matrix.shape[0]
    inverse_matrix = np.array(a0_inv)
    current_matrix = np.array(matrix[0])
    for i in range(1, n):
        a = matrix[i]
        z = calculate_z(current_matrix, inverse_matrix)
        current_matrix = add_row(current_matrix, a)
        denum = np.dot(np.dot(a.T, z), a)
        if np.abs(denum) < EPS:
            inverse_matrix = append_zero_case(inverse_matrix, a, z)
        else:
            inverse_matrix = append_to_inverse(inverse_matrix, a, z, denum)

    return inverse_matrix
