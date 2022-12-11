import numpy as np


def jacobi(A, b, max_iterations=1000, x=None, tol=1e-10):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(A[0]))

    D = np.diag(A)  # Diagonal of A
    R = A - np.diagflat(D)  # Remaining of A after removing Diagonal
    flops = 1

    # Iterate until tol < 1E-10 or i=max iterations
    for i in range(max_iterations):
        x_old = x
        x = (b - np.dot(R, x)) / D
        flops = flops + 3

        if np.allclose(x_old, x, atol=tol, rtol=0.):
            return 0, x, i, flops

    return 1, x, i, flops


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
# indicatorJ, xJ, iterationsJ, flopsJ = jacobi(A, b, maxIt)
# indicatorCG, xCG, iterationsCG, flopsCG = conjugate_gradient(A, bcg, tol, maxIt)
# indicatorSOR, xSOR, iterationsSOR, flopsSOR = SOR(A, b, maxIt, tol)

# print(jacobi(A, b, maxIt))