import numpy as np
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from gaussian_elimination import gaussian_elimination
from steepest_descent import steepest_descent
from utils import create_matrix, CG_COLOR, SD_COLOR, GE_COLOR

# dimensions to test
DIMENSIONS = [10, 20, 50, 100, 200, 300, 500]
# DIMENSIONS = [10, 20, 50, 100, 200, 300]

# testing CG versus gaussian elimination
condition_numbers = [3, 5, 8, 10, 12, 15]
NUM_SAMPLES = 10

# store the average time for each dimension
cg_exec_times = []
cg_exec_times_2 = []
sd_exec_times = []
gauss_exec_times = []

# store the average number of iterations for each dimension
cg_avg_iterations = []
cg_avg_iterations_2 = []
sd_avg_iterations = []

# store norms decreasing
cg_avg_norms = []
cg_avg_norms_2 = []
sd_avg_norms = []

for d in DIMENSIONS:

    print('-----------------------------')
    print(f'dimension: {d}')

    # do every experiment 10 times and take the average
    # first index is time, second is iterations
    cg_sum_stats = [0, 0, 0]
    cg_sum_stats_2 = [0, 0, 0]
    sd_sum_stats = [0, 0, 0]
    gauss_sum_stats = 0
    for s in range(NUM_SAMPLES):
        # create random matrices for testing
        print(f'sample: {s}')
        condition_number = np.random.choice(condition_numbers)
        A = create_matrix(d, condition_number)
        print(f'condition # = {np.linalg.cond(A)}')

        # randomly choose RHS and initial guess
        b = np.random.rand(d, 1)
        x0 = np.random.rand(d, 1)

        # err_code, cg_x_star, steps, cg_iterations, cg_time, cg_norms = conjugate_gradient(A, b, x0, 10e-6, len(A) * 2)
        err_code, cg_x_star_2, steps, cg_iterations_2, cg_time_2, cg_norms_2 = conjugate_gradient_2(A, b, 10e-6, len(A) * 2)

        err_code, sd_x_star, steps, sd_iterations, sd_time, sd_norms = steepest_descent(A, b, x0, 10e-6, len(A) * 2)

        x, gauss_time = gaussian_elimination(A, b)

        # cg_sum_stats[0] += cg_time
        # cg_sum_stats[1] += cg_iterations

        cg_sum_stats_2[0] += cg_time_2
        cg_sum_stats_2[1] += cg_iterations_2
        # cg_sum_stats_2[2] += cg_norms_2

        sd_sum_stats[0] += sd_time
        sd_sum_stats[1] += sd_iterations
        # sd_sum_stats[2] += sd_norms

        gauss_sum_stats += gauss_time

    # cg_exec_times.append(cg_sum_stats[0] / NUM_SAMPLES)
    cg_exec_times_2.append(cg_sum_stats_2[0] / NUM_SAMPLES)
    sd_exec_times.append(sd_sum_stats[0] / NUM_SAMPLES)
    gauss_exec_times.append(gauss_sum_stats / NUM_SAMPLES)

    # norms
    cg_avg_norms_2.append(cg_sum_stats_2[2] / NUM_SAMPLES)
    sd_avg_norms.append(sd_sum_stats[2] / NUM_SAMPLES)

    # cg_avg_iterations.append(cg_sum_stats[1] / NUM_SAMPLES)
    cg_avg_iterations_2.append(cg_sum_stats_2[1] / NUM_SAMPLES)
    sd_avg_iterations.append(sd_sum_stats[1] / NUM_SAMPLES)

# Plotting execution times

# plt.plot(DIMENSIONS, cg_exec_times, label='cg', marker='o')
plt.plot(DIMENSIONS, cg_exec_times_2, label='conjugate gradient', linestyle='--', marker='o', c=CG_COLOR)
plt.plot(DIMENSIONS, sd_exec_times, label='steepest descent', linestyle='--', marker='o', c=SD_COLOR)
plt.plot(DIMENSIONS, gauss_exec_times, label='gaussian elimination', linestyle='--', marker='o', c=GE_COLOR)
plt.legend(loc='upper left')
plt.xlabel('Dimension of A')
plt.ylabel('Exec Time in Seconds')
plt.savefig('./plots/cg-vs-gauss-vs-sd-exec-times.png')
plt.cla()

# Plotting iterations
plt.plot(DIMENSIONS, cg_avg_iterations_2, label='conjugate gradient', linestyle='--', marker='o', c=CG_COLOR)
plt.plot(DIMENSIONS, sd_avg_iterations, label='steepest descent', linestyle='--', marker='o', c=SD_COLOR)
plt.legend(loc='upper left')
plt.xlabel('Dimension of A')
plt.ylabel('Iterations')
plt.savefig(f'./plots/cg-vs-sd-iterations.png')

# Plot Norms
