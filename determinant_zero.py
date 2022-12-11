import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix, COLOR_PALETTE

# dimensions to test
# DIMENSIONS = [10, 20, 50, 100, 200, 300, 500]
DIMENSIONS = [5, 20, 10, 50, 100, 200]

NULL_SPACE_PERCENT = [0.01, 0.05, 0.1, 0.2, 0.5]

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

        eigen = np.linspace(1, 1, d)  # note that the condition number here is inconsequential with custom eigen vals
        print(f'num zero eigen values = {int(len(eigen) * p)}')
        for i in range(int((1 - p) * len(eigen)), len(eigen)):
            eigen[i] = 100_000 * d
            # eigen[i] = 0

        cg_failure_rate = 0
        for s in range(NUM_SAMPLES):
            # create random matrices for testing

            A = create_matrix(d, 0, custom_eigen_vals=eigen)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_failure_rate += err_code

        failure_rates.append(
            [cg_failure_rate / NUM_SAMPLES, d, f'{int(p*100)}%']
        )


COLUMNS = ['CG Failure Rate', 'System Dimension', 'Zero Eigen Values as % of System']
df = pd.DataFrame(failure_rates, columns=COLUMNS)
sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar", palette=COLOR_PALETTE)

# plt.savefig(f'./plots/determinant-zero.png')
plt.savefig(f'./plots/big-eigenvalues.png')
plt.show()
plt.cla()
