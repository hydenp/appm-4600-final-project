import numpy as np

from gcm import conjugate_gradient


def display_results(err_code, xk, steps, iters):
    print(f'solution in {iters} iterations with error code {err_code}')
    print(xk)
    print('here are the steps on the way')
    for s in steps:
        print(s)


# when A is ill-conditioned
# TODO: construct ill-conditioned matrix
A = np.array([[4, 3, 0],
              [3, 4, -1],
              [0, -1, 4]])

b = np.array([[24],
              [30],
              [-24]])

x0 = np.array([[0, 0, 0]])
x0 = x0.T

# err_code, x_star, steps, iters = conjugate_gradient(A, b, x0, 1e-6, 100)
# display_results(err_code, x_star, steps, iters)

print()

# when det(A) = 0
# TODO: setup problem with det(A) that is Positive Definite

# err_code, x_star, steps, iters = conjugate_gradient(A, b, x0, 1e-6, 100)
# display_results(err_code, x_star, steps, iters)


print()

# when A is not symmetric
print('A not symmetric')
A = np.array([[4, 3, 5],
              [3, 4, -1],
              [0, -1, 4]])
b = np.array([[24],
              [30],
              [-24]])

x0 = np.array([[0, 0, 0]])
x0 = x0.T

err_code, x_star, steps, iters = conjugate_gradient(A, b, x0, 1e-6, 100)
display_results(err_code, x_star, steps, iters)
