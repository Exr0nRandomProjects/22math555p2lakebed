from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np

from itertools import combinations

CHARGE_LINES = {    # constant
    0: (1, [(0, 0), (0, 10), (10, 10), (10, 0)]),
    5: (2, [(5, 2), (8, 5), (5, 8), (2, 5)]),
    10: (3, [(5, 5), (6, 6)]),
}

points_pos = np.array(pos)
points_y_pos = []


if __name__ == '__main__':
    print("hello world")

    plt_curves = { d: plt.plot([x for x, _ in arr] + [arr[0][0]], [y for _, y in arr] + [arr[0][1]]) for d, (c, arr )in CHARGE_LINES.items() }

    plt.show()

    for _ in trange(100):
        # inter-line forces
        for (c_charge, c_points), (f_charge, f_points) in combinations(CHARGE_LINES.values(), r=2):
            print(c_charge, f_charge)

    def update_line(hl, new_data):
        hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
        hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
        plt.draw()
