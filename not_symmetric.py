import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix, COLOR_PALETTE

# dimensions to test
DIMENSIONS = [10, 20, 50, 100, 200, 300, 500]
# DIMENSIONS = [5, 10, 20, 50, 100, 200]
# DIMENSIONS = [5, 10, 20, 50]
DISTANCE_MULTIPLIERS = [2, 5, 10, 25, 50]

NUM_SAMPLES = 10  # Number of samples run through for each case

NUM_REPLACED_VALS = 4  # however many values or overwritten
SUFFIX = f'around-diag-change-{NUM_REPLACED_VALS}'
# SUFFIX = f'in-corner-change-{NUM_REPLACED_VALS}'

failure_rates = []
performance_vs = []
iterations_vs = []
iterations_until_exit = []  # Not

for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    condition_number = 200
    print(f'condition number: {condition_number}')

    iterations_until_failure_per_dist = []
    for dist in DISTANCE_MULTIPLIERS:

        cg_failure_rate = 0
        num_iterations = 0

        # for comparison with successful
        proportionate_exec_time_increase_sym_vs_non_sym = []
        proportionate_iters_increase_sym_vs_non_sym = []

        for s in range(NUM_SAMPLES):
            # create random matrices for testing
            A = create_matrix(d, condition_number)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)
            x0 = np.random.rand(d, 1)

            A_save = A  # save a copy of A for running symmetric case

            # set one value to a constant, so it's not symmetric anymore

            # bottom corner
            # A[len(A) - 1][0] *= dist
            # A[len(A) - 1][1] *= dist
            # A[len(A) - 2][0] *= dist
            # A[len(A) - 2][1] *= dist

            # middle
            ABOVE = False

            A[len(A)//2 - 1 - ABOVE][len(A)//2 - 2 + ABOVE] *= dist
            A[len(A)//2 - ABOVE][len(A)//2 - 1 + ABOVE] *= dist
            A[len(A)//2 + 1 - ABOVE][len(A)//2 + ABOVE] *= dist
            A[len(A)//2 + 2 - ABOVE][len(A)//2 + 1 + ABOVE] *= dist

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_failure_rate += err_code
            num_iterations += iterations

            # if non-symmetric found a solution, find on symmetric and compare differences of iterations
            if err_code == 0:
                err_code_s, x_star_s, steps_s, iterations_s, cg_time_s, cg_norms_s = conjugate_gradient_2(A_save, b,
                                                                                                          10e-2,
                                                                                                          len(A) * 2)

                proportionate_exec_time_increase_sym_vs_non_sym.append(cg_time / cg_time_s)
                proportionate_iters_increase_sym_vs_non_sym.append(iterations / iterations_s)

        if len(proportionate_exec_time_increase_sym_vs_non_sym) > 0:
            performance_vs.append([sum(proportionate_exec_time_increase_sym_vs_non_sym) / len(
                proportionate_exec_time_increase_sym_vs_non_sym) - 1, d, dist])
            iterations_vs.append([sum(proportionate_iters_increase_sym_vs_non_sym) / len(
                proportionate_iters_increase_sym_vs_non_sym), d, dist])

        failure_rates.append([cg_failure_rate / NUM_SAMPLES, d, dist])
        iterations_until_failure_per_dist.append(num_iterations / NUM_SAMPLES)

    iterations_until_exit.append(iterations_until_failure_per_dist)

print(performance_vs)
print(iterations_vs)
print(iterations_until_exit)

# plot the results
COLUMNS = ['CG Failure Rate', 'System Dimension', 'Distance Multiplier']
df = pd.DataFrame(failure_rates, columns=COLUMNS)

sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar",
            palette=COLOR_PALETTE)
plt.savefig(f'./plots/not-symmetric-{SUFFIX}.png')
plt.show()
plt.cla()

###
COLUMNS = ['Proportionate Exec Time Increase vs Sym Case', 'System Dimension', 'Distance Multiplier']
df = pd.DataFrame(performance_vs, columns=COLUMNS)
sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar", palette=COLOR_PALETTE)

plt.savefig(f'./plots/not-symmetric-perf-vs-sym-{SUFFIX}.png')
plt.show()
plt.cla()

###
COLUMNS = ['Proportionate # Iterations Increase vs Sym Case', 'System Dimension', 'Distance Multiplier']
df = pd.DataFrame(iterations_vs, columns=COLUMNS)
sns.catplot(data=df, x=COLUMNS[1], y=COLUMNS[0], hue=COLUMNS[2], kind="bar", palette=COLOR_PALETTE)

plt.savefig(f'./plots/not-symmetric-iters-vs-sym-{SUFFIX}.png')
plt.show()
plt.cla()
