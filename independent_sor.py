import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sor import SOR_omega
from utils import create_matrix, COLOR_PALETTE

# dimensions to test
# DIMENSIONS = [20, 10, 50, 100, 200]
DIMENSIONS = [5, 10, 15, 20, 25, 30, 50]

PERCENT_SPARSE = 0.9  # what percent of the diagonals are zero

OMEGA_VALUES = [1, 1.025, 1.03, 1.05, 1.075, 1.1, 1.15, 1.2, 1.5]  # values of omega to test with

NUM_SAMPLES = 10  # the sample size

# variables to store results
failure_rates = []
iterations_per_omega = []
for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    for omega in OMEGA_VALUES:

        sor_failure_rate = 0
        sor_iterations = 0
        for s in range(NUM_SAMPLES):

            # create random matrices for testing
            A = create_matrix(d, 100)
            ZERO_ELEMENTS = len(A) * PERCENT_SPARSE
            for i in reversed(range(3)):
                for j in range(3 - i):
                    A[len(A) - i - 1][j] = 0

            # mirror the elements for symmetry
            A = np.tril(A) + np.tril(A).T - np.diag(np.diag(A))

            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)

            err_code, x_star, num_iterations, flops = SOR_omega(A, np.random.rand(1, d)[0], omega, tol=1e-2)
            # err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            sor_failure_rate += err_code
            sor_iterations += num_iterations

        failure_rates.append(
            [sor_failure_rate / NUM_SAMPLES, d, omega]
        )
        iterations_per_omega.append(
            [sor_iterations / NUM_SAMPLES, d, omega]
        )



COLUMNS = ['CG Failure Rate', 'System Dimension', 'Omega']
df = pd.DataFrame(failure_rates, columns=COLUMNS)
sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar", palette=COLOR_PALETTE)
plt.savefig(f'./plots/independent-sor-percent-sparse-{PERCENT_SPARSE}.png')
plt.cla()


COLUMNS = ['SOR Iterations', 'System Dimension', 'Omega']
df = pd.DataFrame(iterations_per_omega, columns=COLUMNS)
sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar", palette=COLOR_PALETTE)
plt.savefig(f'./plots/independent-sor-iterations-percent-sparse-{PERCENT_SPARSE}.png')
plt.cla()
