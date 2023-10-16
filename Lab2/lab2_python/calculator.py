import numpy as np


def append_ones(image):
    matrix = np.array(image)
    ones = np.ones((1, matrix.shape[1]))
    return np.vstack((matrix, ones))


def to_matrix(image):
    return np.array(image)


def calculate_z(x, x_inv):
    i_matrix = np.identity(x.shape[0])
    return i_matrix - np.dot(x, x_inv)


def calculate_v(n1, n2):
    return np.zeros((n1, n2))


def inverse(x_matrix, method):
    if method == "G":
        return np.zeros(0)
        # Grenvile
    elif method == "M":
        return np.zeros(1)
        # Mure
    else:
        raise ValueError("Unknown method")


def calculate(input_image, output_image, method):
    x_matrix = append_ones(input_image)
    y_matrix = to_matrix(output_image)

    x_inv = inverse(x_matrix, method)
    z = calculate_z(x_matrix, x_inv)
    v = calculate_v(0, 0)
    a_matrix = np.dot(y_matrix, x_inv) + np.dot(v, z.T)
    return a_matrix
