from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange, tqdm
import numpy as np
from numba import njit, vectorize
from numba.typed import List
import numba as nb

from lines import lines as data_lines, points as data_points, depth as data_depth, charges as data_charges

from itertools import combinations, product
from dataclasses import dataclass
import math

@dataclass(eq=False)
class Point:
    pos: np.ndarray
    old_pos: np.ndarray
    tot_vel: nb.float64
    pinned: bool
    def __repr__(self):
        return f"({self.pos[0]:.1f}, {self.pos[1]:.1f})"
spec_Point = [('pos', nb.float64[:]), ('old_pos', nb.float64[:]), ('tot_vel', nb.float64), ('pinned', nb.bool_)]
# from https://github.com/numba/numba/issues/4037#issuecomment-907523015
del Point.__dataclass_params__  # type: ignore
del Point.__dataclass_fields__  # type: ignore
del Point.__match_args__  # type: ignore
Point = nb.experimental.jitclass(spec_Point)(Point)  # type: ignore


@dataclass
class ChargeLine:
    points: List[Point]
    depth: float
    charge: float
    def __repr__(self):
        return str(self.points)

# charge_of_depth = lambda x: 0.01*(x-25)**2 + 7

# @vectorize([float64(Point, Point)])
@njit
def distance(p1: Point, p2: Point):
    return np.linalg.norm(p1.old_pos - p2.old_pos)
@njit
def inter_line_charge(c, d): return 1 * c / max(d, 10)**2
# @njit
# def intra_line_charge(c, d): return c/1.5**d

STRAIGHTNESS_FORCE = 0.01
CHARGE_FORCE_MAX = 100

plt.style.use('dark_background')

@njit
def windowed(iterable, n):
    for i in range(len(iterable)-n+1):
        yield iterable[i:i+n]

def new_Point(x, y, pinned=False):
    return Point(pos=np.array([x, y]), old_pos=np.array([x, y]), tot_vel=0, pinned=pinned)

# INITIAL_CHARGE_LINES = [    # constant
#     ChargeLine(points=List([new_Point(400*math.cos(t), 300*math.sin(t), pinned=True) for t in np.linspace(0, 2*math.pi, 100)]), depth=0, charge=0.5),
#     ChargeLine(points=List([new_Point(375*math.cos(t), 275*math.sin(t)) for t in np.linspace(0, 2*math.pi, 100)]), depth=1, charge=1),
#     ChargeLine(points=List([new_Point(350*math.cos(t), 250*math.sin(t)) for t in np.linspace(0, 2*math.pi, 100)]), depth=1, charge=1.5),
#     ChargeLine(points=List([new_Point(325*math.cos(t), 225*math.sin(t)) for t in np.linspace(0, 2*math.pi, 80)]), depth=2, charge=2),
#     ChargeLine(points=List([new_Point(275*math.cos(t), 200*math.sin(t)) for t in np.linspace(0, 2*math.pi, 80)]), depth=3, charge=1.5),
#     ChargeLine(points=List([new_Point(200*math.cos(t), 180*math.sin(t)) for t in np.linspace(0, 2*math.pi, 60)]), depth=3, charge=1),
#     ChargeLine(points=List([new_Point(100*math.cos(t), 150*math.sin(t), pinned=True) for t in np.linspace(0, 2*math.pi, 30)]), depth=4, charge=1),
# ]


INITIAL_CHARGE_LINES = [
        ChargeLine(List([new_Point(x, y, pinned=(i == 0)) for x, y in line]), depth, charge) for i, (line, depth, charge) in enumerate(zip(data_lines, data_depth, data_charges))
]

for line in INITIAL_CHARGE_LINES:
    print([(int(p.pos[0]), int(p.pos[1])) for p in line.points])

FRICTION = 0.9

def update_points_verlet(charge_lines):
    def kinematics(charge_lines: List[ChargeLine]):
        @njit
        def op_on_cl(points: List[Point]):
            for p in points:
                if p.pinned: continue
                vel = p.pos - p.old_pos
                p.old_pos = p.pos.copy()
                p.pos += vel * FRICTION
                p.tot_vel = 0

        for cl in charge_lines:
            op_on_cl(cl.points)

    def charge_force(charge_lines):
        @njit
        def op(points1, points2, charge):
            for p1 in points1:
                for p2 in points2:
                    if p1.pinned and p2.pinned: continue
                    direction = p2.old_pos - p1.old_pos
                    direction /= np.linalg.norm(direction)

                    force = inter_line_charge(charge, distance(p1, p2))
                    force = min(force, CHARGE_FORCE_MAX)

                    if not p1.pinned:
                        p1.pos -= force * direction; p1.tot_vel += np.linalg.norm(force * direction)
                    if not p2.pinned:
                        p2.pos -= force * direction; p2.tot_vel += np.linalg.norm(force * direction)

        # @njit
        # def intra_line_force(points, charge):
        #     for i, p1 in enumerate(points):
        #         for j, p2 in enumerate(points[i:]):
        #             if i == j or (p1.pinned and p2.pinned): continue
        #             direction = p2.old_pos - p1.old_pos
        #             direction /= np.linalg.norm(direction)
        #
        #             force = intra_line_charge(charge, distance(p1, p2))
        #             force = min(force, CHARGE_FORCE_MAX)
        #
        #             if not p1.pinned:
        #                 p1.pos -= force * direction; p1.tot_vel += np.linalg.norm(force * direction)
        #             if not p2.pinned:
        #                 p2.pos -= force * direction; p2.tot_vel += np.linalg.norm(force * direction)
        #

        for i, cl_a in enumerate(charge_lines):
            for cl_b in charge_lines[i+1:]:
                op(cl_a.points, cl_b.points, cl_a.charge * cl_b.charge)
                # intra_line_force(cl_a.points, cl_a.charge **2)

    def straightness_force(charge_lines):
        @njit
        def apply_straightness(l: Point, c: Point, r: Point):
            v1 = l.old_pos - c.old_pos
            v2 = r.old_pos - c.old_pos
            delta = STRAIGHTNESS_FORCE * (v1+v2)

            c.pos += delta; c.tot_vel += np.linalg.norm(delta)
            if not l.pinned:
                l.pos -= delta/2; l.tot_vel += np.linalg.norm(delta/2)
            if not r.pinned:
                r.pos -= delta/2; r.tot_vel += np.linalg.norm(delta/2)

        @njit
        def op(points):
            for l, c, r in windowed(points, n=3):
                if c.pinned: continue
                apply_straightness(l, c, r)

        for cl in charge_lines:
            if len(cl.points) < 3: continue
            op(cl.points)
            if not cl.points[0].pinned: apply_straightness(cl.points[-1], cl.points[0], cl.points[1])
            if not cl.points[-1].pinned: apply_straightness(cl.points[-2], cl.points[-1], cl.points[0])

    kinematics(charge_lines)
    charge_force(charge_lines)
    straightness_force(charge_lines)
    # print(charge_lines)

fig, (vis_ax, convergence_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
max_vel_data = List()
avg_vel_data = List()
tot_vel_data = List()
max_vel_line, = convergence_ax.plot([], [], label="maximum velocity")
avg_vel_line, = convergence_ax.plot([], [], label="average velocity")
tot_vel_line, = convergence_ax.plot([], [], label="max total velocity")
convergence_ax.set_xlabel('iterations')
convergence_ax.legend()

def animate(step, line_plots, charge_lines):

    update_points_verlet(charge_lines)

    for line, cl in zip(line_plots, charge_lines):
        line.set_xdata([p.pos[0] for p in cl.points] + [cl.points[0].pos[0]])
        line.set_ydata([p.pos[1] for p in cl.points] + [cl.points[0].pos[1]])

    vels = [np.linalg.norm(p.pos - p.old_pos) for cl in charge_lines for p in cl.points]
    tot_vels = [p.tot_vel for cl in charge_lines for p in cl.points]
    max_vel_data.append(max(vels))
    avg_vel_data.append(sum(vels)/len(vels))
    tot_vel_data.append(max(tot_vels))
    max_vel_line.set_ydata(max_vel_data)
    max_vel_line.set_xdata(range(len(max_vel_data)))
    avg_vel_line.set_ydata(avg_vel_data)
    avg_vel_line.set_xdata(range(len(avg_vel_data)))
    tot_vel_line.set_ydata(tot_vel_data)
    tot_vel_line.set_xdata(range(len(tot_vel_data)))
    convergence_ax.set_xlim(0, len(avg_vel_data))
    y_scale = max(max(tot_vel_data), max(max_vel_data))
    # y_scale = max(max_vel_data)
    convergence_ax.set_ylim(-0.2*y_scale, 1.2*y_scale)


if __name__ == '__main__':
    print("hello world")


    plt_curves = [ vis_ax.plot([], [], marker='o', label=f"{cl.depth}: {cl.charge}")[0] for cl in INITIAL_CHARGE_LINES ]
    vis_ax.legend()

    vis_ax.set_xlim(-10, 600)
    vis_ax.set_ylim(-10, 500)

    ani = FuncAnimation(fig, animate, fargs=[plt_curves, INITIAL_CHARGE_LINES], frames=trange(int(1e5)), interval=100, blit=False)



    plt.show()

