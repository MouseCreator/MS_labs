from io_reader import read
from approximation import calculate

import sympy as sp

if __name__ == "__main__":
    mtx_input = read('input\\y10.txt')

    c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1 c2 c3 c4 m1 m2 m3')
    beta_init = {c1: 0.1, c2: 0.1, c3: 0.4}
    beta_init_real = {c1: 0.14, c2: 0.3, c3: 0.2}
    approximated = calculate(mtx_input, {m1: 12, m2: 28, m3: 18, c4: 0.12},
                             [c1, c2, c3], beta_init, 1e-6)
    print("Approximation:", approximated)
