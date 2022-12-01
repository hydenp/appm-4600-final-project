import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from conjugate_gradient import conjugate_gradient
from utils import create_matrix

# dimensions to test
# DIMENSIONS = [10, 50, 100, 200, 300, 500, 1_000, 2_000, 5_000]
# DIMENSIONS = [10, 50, 100, 200, 300, 500]
DIMENSIONS = [10, 50, 100, 200, 300]

condition_numbers = [100, 1_000, 10_000, 50_000, 100_000]
NUM_SAMPLES = 10

exec_times = {}
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
            A = create_matrix(d, c)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)
            x0 = np.random.rand(d, 1)

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient(A, b, x0, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_sum_times += cg_time
            cg_success_rate += (err_code == 1)

        cond_number_to_cg_exec_times[c * d] = {
            'time': cg_sum_times / NUM_SAMPLES,
            'success_rate': cg_success_rate / NUM_SAMPLES
        }

    exec_times[d] = cond_number_to_cg_exec_times

plt.legend(loc='upper left')
plt.title(f'Conjugate Gradient with ill-conditioned systems')
plt.xlabel('Dimension')
plt.ylabel('Exec Time in Seconds')

plt.savefig('cg-vs-gauss.png')
plt.show()
