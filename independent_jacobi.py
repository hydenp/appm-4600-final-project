import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from sor import SOR, SOR_omega
from jacobi import jacobi
from utils import create_matrix, COLOR_PALETTE

# dimensions to test
# DIMENSIONS = [10, 20, 50, 100, 200, 300, 500]
# DIMENSIONS = [20, 10, 50, 100, 200]
DIMENSIONS = [10, 20, 30, 50, 80, 100]

# NULL_SPACE_PERCENT = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]
NULL_SPACE_PERCENT = [0.1, 0.2, 0.5, 0.7, 0.8]

NUM_SAMPLES = 10

# TODO: issue is there is overflow but numpy does not warn when this happens in arrays
# need to look into some sort of check so we can catch this and make sure
# we catch this an unsuccessful

failure_rates = []
for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    for p in NULL_SPACE_PERCENT:

        cg_failure_rate = 0
        for s in range(NUM_SAMPLES):
            # create random matrices for testing

            A = create_matrix(d, 100)
            ZERO_ELEMENTS = len(A) * p
            for i in reversed(range(3)):
                for j in range(3 - i):
                    A[len(A) - i - 1][j] = 0

            # mirror the elements
            A = np.tril(A) + np.tril(A).T - np.diag(np.diag(A))

            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)

            err_code, x_star, num_iterations, flops = jacobi(A, np.random.rand(1, d)[0], 1_000)
            # err_code, x_star, num_iterations, flops = SOR(A, np.random.rand(1, d)[0], tol=1e-2)
            # err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_failure_rate += err_code

        failure_rates.append(
            [cg_failure_rate / NUM_SAMPLES, d, f'{int(p * 100)}%']
        )

COLUMNS = ['CG Failure Rate', 'System Dimension', 'Percent Zero Elements Diag']
df = pd.DataFrame(failure_rates, columns=COLUMNS)
sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar", palette=COLOR_PALETTE)

# plt.savefig(f'./plots/determinant-zero.png')
plt.savefig(f'./plots/independent-jacobi.png')
plt.show()
plt.cla()
