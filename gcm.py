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
def conjugate_gradient(
        A: np.array,
        b: np.array,
        x0: np.array,
        tolerance: float,
        max_iterations: int = None
) -> (np.array, int):

    # define the residual calculation
    def r(x):
        return A @ x - b

    rk = r(x0)
    pk = - rk
    xk = x0
    steps = [list(np.reshape(xk, (len(xk), 1)[0]))]

    iterations = 0
    while np.linalg.norm(rk) > tolerance:

        if max_iterations is not None and iterations == max_iterations:
            return 1, xk, steps, iterations

        alpha = (- rk.T @ pk) / (pk.T @ A @ pk)
        xk = xk + alpha * pk
        rk = A @ xk - b
        beta = (rk.T @ A @ pk) / (pk.T @ A @ pk)
        pk = - r(xk) + beta * pk

        steps.append(list(np.reshape(xk, (len(xk), 1)[0])))
        iterations += 1

    return 0, xk, steps, iterations


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

    err_code, x_star, steps, iters = conjugate_gradient(A, b, x0, 1e-10)
    print(x_star)
    print(iters)

    # plotting the steps
    xs = [x[0] for x in steps]
    ys = [y[1] for y in steps]
    zs = [y[2] for y in steps]

    fig = plt.figure(figsize=(12, 12), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='x', linestyle='--', color='red')

    for x, y, z in zip(xs, ys, zs):
        label = f'({x:.2f}, {y:.2f}, {z:.2f})'
        ax.text(x, y, z, label)

    ax.set_title("CG Steps")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    # plt.show()
    plt.savefig('3d-plot.png', bbox_inches='tight')
