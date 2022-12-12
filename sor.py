import numpy as np


def SOR(A, b, max_iterations=1_000, tol=1e-6):
    x = np.zeros_like(b)
    iterations = 0
    flops = 0
    for it_count in range(1, max_iterations):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            flops = flops + 5

        if np.allclose(x, x_new, atol=tol, rtol=0.):
            if np.allclose(A @ x, b, atol=tol, rtol=0.):
                return 0, x, iterations, flops

        x = x_new
        iterations = iterations + 1
    return 1, x, iterations, flops


def SOR_omega(A, b, omega, max_iterations=100, tol=1e-6):
    x = np.zeros_like(b)
    flops = 0
    OmegaPrime = (1 - omega)
    for iteration in range(1, max_iterations):
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * x[j]
                    flops = flops + 2
                x[i] = OmegaPrime * x[i] + (omega / A[i][i]) * (b[i] - sigma)
                flops = flops + 5
            error = np.linalg.norm(np.matmul(A, x) - b)
            if error < tol:
                return 0, x, iteration, flops
    return 1, x, iteration, flops


# System
A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])

b = np.array([6., 25., -11., 15.])

bcg = np.array([[6.],
                [25.],
                [-11.],
                [15.]])

maxIt = 1000
tol = 1e-10

# Iteration Calls and Errors
# indicatorJ, xJ, iterationsJ, flopsJ = jacobi2(A, b, maxIt)
# indicatorCG, xCG, iterationsCG, flopsCG = conjugate_gradient(A, bcg, tol, maxIt)
# indicatorSOR, xSOR, iterationsSOR, flopsSOR = SOR(A, b, maxIt, tol)
