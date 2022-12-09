import numpy as np
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from steepest_descent import steepest_descent
from utils import CG_COLOR, SD_COLOR

if __name__ == '__main__':

    # Problem formulation
    A = np.array([[4, 3, 0],
                  [3, 4, -1],
                  [0, -1, 4]])
    b = np.array([[24],
                  [30],
                  [-24]])

    x0 = np.array([[0, 0, 0]])
    x0 = x0.T

    err_code, x_star, cg_steps, iters, proc_time, cg_norms = conjugate_gradient_2(A, b, 1e-10, 1_000)
    error_code, x_star, sd_steps, iterations, sd_time, sd_norms = steepest_descent(A, b, x0, 10e-10, 100)

    # plotting the steps
    cg_xs = [x[0] for x in cg_steps]
    cg_ys = [y[1] for y in cg_steps]
    cg_zs = [y[2] for y in cg_steps]

    sd_xs = [x[0] for x in sd_steps]
    sd_ys = [y[1] for y in sd_steps]
    sd_zs = [y[2] for y in sd_steps]

    fig = plt.figure(figsize=(12, 12), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cg_xs, cg_ys, cg_zs, marker='x', linestyle='--', color=CG_COLOR, label='conjugate gradient')
    ax.plot(sd_xs, sd_ys, sd_zs, marker='o', linestyle='--', color=SD_COLOR, label='steepest descent')
    ax.legend(loc='upper left')

    # add labels for CG points
    for x, y, z in zip(cg_xs, cg_ys, cg_zs):
        label = f'({x:.2f}, {y:.2f}, {z:.2f})'
        ax.text(x, y, z, label)

    # for x, y, z in zip(sd_xs, sd_ys, sd_zs):
    #     label = f'({x:.2f}, {y:.2f}, {z:.2f})'
    #     ax.text(x, y, z, label)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    # plt.show()
    plt.savefig('./plots/3d-cg-sd-plot.png', bbox_inches='tight')
    plt.cla()

    ############################################################################################
    # Remove the first couple steps as those are significantly larger
    # and make last few of sd harder to see
    REMOVED_STEPS = 3

    cg_xs = [x[0] for x in cg_steps]
    cg_ys = [y[1] for y in cg_steps]
    cg_zs = [y[2] for y in cg_steps]

    sd_xs = [x[0] for x in sd_steps]
    sd_ys = [y[1] for y in sd_steps]
    sd_zs = [y[2] for y in sd_steps]

    cg_xs = cg_xs[REMOVED_STEPS:]
    cg_ys = cg_ys[REMOVED_STEPS:]
    cg_zs = cg_zs[REMOVED_STEPS:]

    sd_xs = sd_xs[REMOVED_STEPS:]
    sd_ys = sd_ys[REMOVED_STEPS:]
    sd_zs = sd_zs[REMOVED_STEPS:]

    fig = plt.figure(figsize=(12, 12), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cg_xs, cg_ys, cg_zs, marker='x', linestyle='--', color=CG_COLOR, label='conjugate gradient')
    ax.plot(sd_xs, sd_ys, sd_zs, marker='o', linestyle='--', color=SD_COLOR, label='steepest descent')
    ax.legend(loc='upper left')

    for x, y, z in zip(cg_xs, cg_ys, cg_zs):
        label = f'({x:.2f}, {y:.2f}, {z:.2f})'
        ax.text(x, y, z, label)

    # for x, y, z in zip(sd_xs, sd_ys, sd_zs):
    #     label = f'({x:.2f}, {y:.2f}, {z:.2f})'
    #     ax.text(x, y, z, label)

    # ax.set_title("CG and Steps")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    # plt.show()
    plt.savefig(f'./plots/3d-cg-sd-plot-remove-{REMOVED_STEPS}-steps.png', bbox_inches='tight')


