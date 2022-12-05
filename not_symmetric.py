import numpy as np
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix

# dimensions to test
# DIMENSIONS = [10, 50, 100, 200, 300, 500, 1_000, 2_000, 5_000]
DIMENSIONS = [10, 50, 100, 200, 300, 500]
# DIMENSIONS = [5, 10, 50, 100, 200]
# DISTANCES = [1, 10, 100, 1_000, 1_000_000]
DISTANCES = [1, 10, 100, 1_000]

NUM_SAMPLES = 10

success_rates = []
for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    condition_number = 100
    print(f'condition number: {condition_number}')

    success_rates_per_dist = []
    for dist in DISTANCES:
        cg_success_rate = 0
        for s in range(NUM_SAMPLES):
            # create random matrices for testing
            A = create_matrix(d, condition_number)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)
            x0 = np.random.rand(d, 1)

            # set one value to a constant, so it's not symmetric anymore
            A[len(A) - 1][0] = dist * len(A)
            A[len(A) - 1][1] = dist * len(A)
            A[len(A) - 1][2] = dist * len(A)
            A[len(A) - 2][0] = dist * len(A)
            A[len(A) - 2][1] = dist * len(A)
            A[len(A) - 2][2] = dist * len(A)

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            # print(steps)
            print(f'error code: {err_code}')

            cg_success_rate += (err_code == 0)

        success_rates_per_dist.append(cg_success_rate / NUM_SAMPLES)
    success_rates.append(success_rates_per_dist)

print(success_rates)
exit()

plt.legend(loc='upper left')
plt.title(f'Conjugate Gradient with ill-conditioned systems')
plt.xlabel('Dimension')
plt.ylabel('Exec Time in Seconds')

plt.savefig('cg-vs-gauss.png')
plt.show()
