import numpy as np
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix

# dimensions to test
# DIMENSIONS = [10, 50, 100, 200, 300, 500, 1_000, 2_000, 5_000]
# DIMENSIONS = [10, 50, 100, 200, 300, 500]
DIMENSIONS = [5, 10, 50, 100, 200, 300]

NUM_SAMPLES = 10

# TODO: issue is there is overflow but numpy does not warn when this happens in arrays
# need to look into some sort of check so we can catch this and make sure
# we catch this an unsuccessful

success_rates = []
for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    cg_success_rate = 0
    condition_number = 100
    print(f'condition number: {condition_number}')
    for s in range(NUM_SAMPLES):
        # create random matrices for testing

        eigen = np.linspace(1, condition_number, d)
        eigen[len(eigen) // 2:] = np.array([0]) * (len(eigen) // 2)
        A = create_matrix(d, condition_number, custom_eigen_vals=eigen)
        print(f'condition # = {np.linalg.cond(A)}')

        # always start at zero x0 = np.random.rand(d, 1)
        b = np.random.rand(d, 1)

        # set one value to a constant, so it's not symmetric anymore
        A[len(A) - 1][len(A) - 1] = 0

        err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
        print(f'error code: {err_code}')

        cg_success_rate += (err_code == 0)

    success_rates.append(
        cg_success_rate / NUM_SAMPLES
    )


print(success_rates)
exit()

plt.legend(loc='upper left')
plt.title(f'Conjugate Gradient with ill-conditioned systems')
plt.xlabel('Dimension')
plt.ylabel('Exec Time in Seconds')

plt.savefig('cg-vs-gauss.png')
plt.show()
