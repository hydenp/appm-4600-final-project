import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from conjugate_gradient import conjugate_gradient_2
from utils import create_matrix

# dimensions to test
# DIMENSIONS = [10, 50, 100, 200, 300, 500, 1_000, 2_000, 5_000]
# DIMENSIONS = [10, 50, 100, 200, 300, 500]
# DIMENSIONS = [5, 10, 50, 100, 200, 300]
DIMENSIONS = [5, 10, 50, 100, 200]

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
            eigen[i] = 0

        cg_failure_rate = 0
        for s in range(NUM_SAMPLES):
            # create random matrices for testing

            A = create_matrix(d, 0, custom_eigen_vals=eigen)
            print(f'condition # = {np.linalg.cond(A)}')

            # always start at zero x0 = np.random.rand(d, 1)
            b = np.random.rand(d, 1)
            # b = np.zeros((d, 1))
            # b = np.reshape(eigen, (len(eigen), 1))

            err_code, x_star, steps, iterations, cg_time, cg_norms = conjugate_gradient_2(A, b, 10e-2, len(A) * 2)
            print(f'error code: {err_code}')

            cg_failure_rate += err_code

        failure_rates.append(
            [cg_failure_rate / NUM_SAMPLES, d, int(p*100)]
        )


df = pd.DataFrame(failure_rates, columns=['Failure Rate', 'System Dimension', '% Zero Eigen Values'])

sns.color_palette("hls", 8)
sns.catplot(data=df, x="System Dimension", y="Failure Rate", hue="% Zero Eigen Values", kind="bar",
            palette=sns.color_palette("flare"))
plt.show()
plt.savefig('determinant-zero.png')

plt.show()
