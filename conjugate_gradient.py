import time

import numpy as np
from matplotlib import pyplot as plt


# implement the conjugate-gradient method
# 1. Initialize x₀
# 2. Calculate r₀ = Ax₀ − b
# 3. Assign p₀ = −r₀
# 4. For k = 0, 1, 2, …:
#     * calculate αₖ = -rₖ'pₖ / pₖ'Apₖ
#     * update xₖ₊₁ = xₖ + αₖpₖ
#     * calculate rₖ₊₁ = Axₖ₊₁ - b
#     * calculate βₖ₊₁ = rₖ₊₁'Apₖ / pₖ'Apₖ
#     * update pₖ₊₁ = -rₖ₊₁ + βₖ₊₁pₖ
def conjugate_gradient(A: np.array, b: np.array, x0: np.array, tolerance: float, max_iterations: int = None):
    start_time = time.process_time()

    rk = A @ x0 - b
    pk = - rk
    xk = x0
    steps = [list(np.reshape(xk, (len(xk), 1)[0]))]

    iterations = 0
    while np.linalg.norm(rk) > tolerance:

        if max_iterations is not None and iterations == max_iterations:
            return 1, xk, steps, iterations, time.process_time() - start_time

        A_pk = A @ pk
        pk_A_pk = pk.T @ A_pk

        # alpha = (- rk.T @ pk) / (pk.T @ A_pk)
        alpha = (- rk.T @ pk) / pk_A_pk
        xk = xk + alpha * pk
        rk = A @ xk - b
        # beta = (rk.T @ A_pk) / (pk.T @ A_pk)
        beta = (rk.T @ A_pk) / pk_A_pk
        pk = - rk + beta * pk

        steps.append(list(np.reshape(xk, (len(xk), 1)[0])))
        iterations += 1

    return 0, xk, steps, iterations, time.process_time() - start_time


def conjugate_gradient_2(A: np.array, b: np.array, tolerance: float, max_iterations: int):
    start = time.process_time()
    x0 = np.zeros((len(A), 1))
    r0 = b
    p0 = r0
    steps = [list(np.reshape(x0, (len(x0), 1)[0]))]

    for j in range(max_iterations):
        A_p = A @ p0
        A_norm_p = p0.T @ A_p
        norm_r0 = r0.T @ r0
        alpha = norm_r0 / A_norm_p
        xk = x0 + alpha * p0
        rk = r0 - alpha * A_p

        beta = rk.T @ rk / norm_r0
        pk = rk + beta * p0

        steps.append(list(np.reshape(x0, (len(x0), 1)[0])))

        if np.linalg.norm(xk - x0) / np.linalg.norm(xk) < tolerance:
            return 0, xk, steps, j + 1, time.process_time() - start

        x0 = xk
        p0 = pk
        r0 = rk

    return 0, x0, steps, max_iterations, time.process_time() - start
