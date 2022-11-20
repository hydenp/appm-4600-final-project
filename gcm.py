import numpy as np


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
def conjugate_gradient(A: np.array, b: np.array, x0: np.array, tolerance: float, max_iterations: int = None) -> (np.array, int):

    # define the residual calculation
    def r(x):
        return A @ x - b

    rk = r(x0)
    pk = - rk
    xk = x0
    steps = [xk]

    iterations = 0
    while np.linalg.norm(rk) > tolerance:

        if max_iterations is not None and iterations == max_iterations:
            raise Exception('Max iterations computed without reaching tolerance.')

        alpha = (- rk.T @ pk) / (pk.T @ A @ pk)
        xk = xk + alpha * pk
        rk = A @ xk - b
        beta = (rk.T @ A @ pk) / (pk.T @ A @ pk)
        pk = - r(xk) + beta * pk

        steps.append(xk)
        iterations += 1

    return xk, steps, iterations


if __name__ == '__main__':

    A = np.array([[4, 3, 0],
                  [3, 4, -1],
                  [0, -1, 4]])
    b = np.array([[24],
                  [30],
                  [-24]])

    x0 = np.array([[0, 0, 0]])
    x0 = x0.T

    x_star, steps, iters = conjugate_gradient(A, b, x0, 1e-10)
    print(x_star)
    print(iters)
    print(A @ x_star)
