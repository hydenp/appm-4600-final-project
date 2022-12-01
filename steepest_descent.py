import time

import numpy as np


def steepest_descent(A: np.array, b: np.array, x0: np.array, tolerance: float, max_iterations: int):
    start = time.process_time()
    steps = [list(np.reshape(x0, (len(x0), 1)[0]))]
    norms = []
    for j in range(max_iterations):
        rk = b - A @ x0
        alpha = (rk.T @ rk) / (rk.T @ A @ rk)
        xk = x0 + alpha * rk

        # add to the steps and norms
        steps.append(list(np.reshape(xk, (len(xk), 1)[0])))
        norms.append(np.linalg.norm(rk))

        if np.linalg.norm(xk - x0) / np.linalg.norm(xk) < tolerance:
            return 0, xk, steps, j + 1, time.process_time() - start, norms

        x0 = xk

    return 1, x0, steps, max_iterations, time.process_time() - start, norms


if __name__ == '__main__':
    # Problem formulation
    A = np.array([[4, 3, 0],
                  [3, 4, -1],
                  [0, -1, 4]])
    b = np.array([[24],
                  [30],
                  [-24]])

    x0 = np.array([[0, 0, 0]])
    x0 = x0.T

    error_code, x_star, steps, iterations, sd_time, sd_norms = steepest_descent(A, b, x0, 10e-10, 100)
    print(x_star)
    print(iterations)
