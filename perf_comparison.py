import numpy as np
import scipy

from gcm import conjugate_gradient

#

A = np.array([[4, 3, 0],
              [3, 4, -1],
              [0, -1, 4]])


def f(x):
    return A @ x


def j(x):
    return np.array([[4, 0, 0],
                     [0, 4, 0],
                     [0, 0, 4]]) @ x


b = np.array([[24],
              [30],
              [-24]])

x0 = np.array([[0, 0, 0]])
x0 = x0.T


print(conjugate_gradient(A, b, x0, 10e-6, 100))

# Gaussian Elimination


scipy.linalg.lu_factor(A)
# compare with Gradient Descent


# compare with Newton-Raphson
