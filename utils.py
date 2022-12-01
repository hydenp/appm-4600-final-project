import numpy as np


# https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def create_matrix(dimension, desired_cond, custom_eigen_vals=None):
    # create random orthonormal matrix
    A = rvs(dimension)

    # create matrix with the desired condition number

    if custom_eigen_vals is not None:
        assert len(custom_eigen_vals) == dimension
        return A.T @ np.diag(custom_eigen_vals) @ A

    eigen = np.linspace(1, desired_cond, dimension)
    return A.T @ np.diag(eigen) @ A
