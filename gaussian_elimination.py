import time

from scipy import linalg


def gaussian_elimination(A, b):
    """
    gaussian elimination using LAPACK.
    """
    start = time.process_time()

    # factor step
    lu, piv, info = linalg.lapack.dgetrf(A)

    # solve step
    x, _ = linalg.lapack.dgetrs(lu, piv, b)

    return x, time.process_time() - start
