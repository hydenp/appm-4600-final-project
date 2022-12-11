import numpy as np
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from utils import CG_COLOR

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

    err_code, x_star, steps, iters, proc_time, cg_norms = conjugate_gradient_2(A, b, 1e-10, 1_000)
    print(x_star)
    print(iters)

    # plotting the steps
    xs = [x[0] for x in steps]
    ys = [y[1] for y in steps]
    zs = [y[2] for y in steps]

    fig = plt.figure(figsize=(12, 12), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='x', linestyle='--', color=CG_COLOR)

    for x, y, z in zip(xs, ys, zs):
        label = f'({x:.2f}, {y:.2f}, {z:.2f})'
        ax.text(x, y, z, label)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    # plt.show()
    plt.savefig('./plots/3d-plot.png', bbox_inches='tight')
