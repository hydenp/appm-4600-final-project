import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix, COLOR_PALETTE

# dimensions to test
# DIMENSIONS = [10, 50, 100, 200, 300, 500, 1_000, 2_000, 5_000]
# DIMENSIONS = [10, 50, 100, 200, 300, 500]
DIMENSIONS = [10, 50, 100, 200, 300]

# condition_numbers = [100, 1_000, 10_000, 50_000, 100_000]
condition_numbers = [2, 5, 10, 50, 100, 300, 500, 1_000]
NUM_SAMPLES = 10

failure_rates = []
for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    cond_number_to_cg_exec_times = {}
    for c in condition_numbers:
        cg_sum_times = 0
        cg_success_rate = 0
        print(f'condition number: {c * d}')
        for s in range(NUM_SAMPLES):
            # create random matrices for testing
            A = create_matrix(d, c*d)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)
            x0 = np.random.rand(d, 1)

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_success_rate += err_code

        failure_rates.append(
            [cg_success_rate / NUM_SAMPLES, d, c]
        )


df = pd.DataFrame(failure_rates, columns=['Failure Rate', 'System Dimension', 'Condition Number multiplier'])

sns.catplot(data=df, x="System Dimension", y="Failure Rate", hue="Condition Number multiplier", kind="bar",
            palette=COLOR_PALETTE)

plt.savefig('./plots/ill-conditioned-systems-multiplier.png')
plt.show()
