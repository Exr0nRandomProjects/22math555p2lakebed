from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange, tqdm
import numpy as np

from itertools import combinations, product
from dataclasses import dataclass
from typing import List
import math

@dataclass
class Point:
    pos: np.ndarray
    old_pos: np.ndarray
    pinned: bool = False

@dataclass
class ChargeLine:
    points: List[Point]
    depth: float
    charge: float

charge_of_depth = lambda x: 0.01*(x-25)**2 + 7
def distance(p1: Point, p2: Point):
    return np.linalg.norm(p1.old_pos - p2.old_pos)
inter_line_charge = lambda c, d: 0.01 * c / d**4
intra_line_charge = lambda c, d: c/1.5**d
CHARGE_CONSTANT = 0.001

def new_Point(x, y, pinned=False):
    return Point(pos=np.array([x, y]), old_pos=np.array([x, y]), pinned=pinned)

INITIAL_CHARGE_LINES = [    # constant
    ChargeLine(points=[new_Point(18*math.cos(t), 15*math.sin(t), pinned=True) for t in np.linspace(0, 2*math.pi, 50)], depth=0, charge=1),
    ChargeLine(points=[new_Point(5*math.cos(t), 3*math.sin(t)) for t in np.linspace(0, 2*math.pi, 50)], depth=1, charge=1),
    ChargeLine(points=[new_Point(4*math.cos(t), 2.5*math.sin(t)) for t in np.linspace(0, 2*math.pi, 50)], depth=2, charge=1.2),
    ChargeLine(points=[new_Point(3*math.cos(t), 2*math.sin(t)) for t in np.linspace(0, 2*math.pi, 40)], depth=3, charge=1),
    ChargeLine(points=[new_Point(2*math.cos(t), 1.5*math.sin(t), pinned=True) for t in np.linspace(0, 2*math.pi, 40)], depth=4, charge=1),
]

FRICTION = 0.9

def update_points_verlet(charge_lines):
    def kinematics(charge_lines):
        for cl in charge_lines:
            for p in cl.points:
                if p.pinned: continue
                vel = p.pos - p.old_pos
                p.old_pos = p.pos.copy()
                p.pos += vel * FRICTION

    def charge_force(charge_lines):
        # for cl_a, cl_b in product(charge_lines):
        for i, cl_a in enumerate(charge_lines):
            for cl_b in charge_lines[i+1:]:
                for p1 in cl_a.points:
                    for p2 in cl_b.points:
                        if p1.pinned and p2.pinned: continue
                        direction = p2.old_pos - p1.old_pos
                        direction /= np.linalg.norm(direction)

                        force = CHARGE_CONSTANT * cl_a.charge* cl_b.charge / distance(p1, p2)**4

                        if not p1.pinned: p1.pos -= force * direction
                        if not p2.pinned: p2.pos += force * direction

    kinematics(charge_lines)
    charge_force(charge_lines)
    # print(charge_lines)

def animate(step, line_plots, charge_lines, pbar):
    pbar.set_description(f"step = {step}")
    pbar.update(1)
    print("hawo")

    update_points_verlet(charge_lines)

    for line, cl in zip(line_plots, charge_lines):
        line.set_xdata([p.pos[0] for p in cl.points] + [cl.points[0].pos[0]])
        line.set_ydata([p.pos[1] for p in cl.points] + [cl.points[0].pos[1]])

if __name__ == '__main__':
    print("hello world")


    fig, (vis_ax, convergance_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    plt_curves = [ vis_ax.plot([], [])[0] for _ in INITIAL_CHARGE_LINES ]

    vis_ax.set_xlim(-10, 10)
    vis_ax.set_ylim(-8, 8)

    with tqdm() as pbar:
        ani = FuncAnimation(fig, animate, fargs=[plt_curves, INITIAL_CHARGE_LINES, pbar], frames=60, interval=100, blit=False)



    plt.show()

    # for _ in trange(100):
    #     # inter-line forces
    #     for (c_charge, c_points), (f_charge, f_points) in combinations(CHARGE_LINES.values(), r=2):
    #         print(c_charge, f_charge)
    #
    # def update_line(hl, new_data):
    #     hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    #     hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    #     plt.draw()
