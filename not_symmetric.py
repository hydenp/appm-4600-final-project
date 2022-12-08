import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix

# dimensions to test
# DIMENSIONS = [10, 50, 100, 200, 300, 500, 1_000, 2_000, 5_000]
# DIMENSIONS = [10, 50, 100, 200, 300, 500]
DIMENSIONS = [5, 10, 50, 100, 200]
DISTANCES = [2, 5, 10, 25, 50]

NUM_SAMPLES = 10

failure_rates = []
performance = []
iterations_until_failure = []
for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    condition_number = 500
    print(f'condition number: {condition_number}')

    failure_rates_per_dist = []
    iterations_until_failure_per_dist = []
    for dist in DISTANCES:
        cg_failure_rate = 0
        num_iterations = 0
        for s in range(NUM_SAMPLES):
            # create random matrices for testing
            A = create_matrix(d, condition_number)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)
            x0 = np.random.rand(d, 1)

            # set one value to a constant, so it's not symmetric anymore
            A[len(A) - 1][0] *= dist
            A[len(A) - 1][1] *= dist
            A[len(A) - 1][2] *= dist
            A[len(A) - 2][0] *= dist
            A[len(A) - 2][1] *= dist
            A[len(A) - 2][2] *= dist

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_failure_rate += err_code
            num_iterations += iterations

        failure_rates.append([cg_failure_rate / NUM_SAMPLES, d, dist])
        iterations_until_failure_per_dist.append(num_iterations / NUM_SAMPLES)
    iterations_until_failure.append(iterations_until_failure_per_dist)

print(iterations_until_failure)

# load in the results from running cg
df = pd.DataFrame(failure_rates, columns=['Failure Rate', 'System Dimension', 'Distance Multiplier'])

sns.color_palette("hls", 8)
sns.catplot(data=df, x="System Dimension", y="Failure Rate", hue="Distance Multiplier", kind="bar",
            palette=sns.color_palette("flare"))
plt.savefig('not-symmetric-change-6.png')
plt.show()
