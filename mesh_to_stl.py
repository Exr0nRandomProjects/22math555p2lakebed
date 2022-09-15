import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import matplotlib.animation



from dataclasses import dataclass

GRAVITY = np.array([0, -0.1])
FRICTION = 0.9999

NUM_RESOLVE_STEPS = 8

N_COLS = 12
N_ROWS = 12

@dataclass
class Point:
    pos: np.ndarray
    old_pos: np.ndarray
    pinned: bool

@dataclass
class Stick:
    p_a: Point
    p_b: Point
    len: float

def distance(p1: Point, p2: Point):
    return np.linalg.norm(p1.pos - p2.pos)

points = [
    Point(pos=np.array([x, y]), old_pos=np.array([x, y]), pinned=False)
    for x in np.linspace(-10, 1, N_COLS) for y in np.linspace(-10, 1, N_ROWS) ]

for i, p in enumerate(points):
    if i % N_ROWS == N_ROWS-1 and np.random.random() < 0.3:
        p.pinned = True

def make_stick_from_indicies(c1, r1, c2, r2):
    p1 = points[c1*N_ROWS + r1]
    p2 = points[c2*N_ROWS + r2]
    return Stick(p_a=p1, p_b=p2, len=distance(p1, p2))

sticks = [ make_stick_from_indicies(c, r, c+1, r) for c in range(N_COLS-1) for r in range(N_ROWS) ] \
        + [ make_stick_from_indicies(c, r, c, r+1) for c in range(N_COLS) for r in range(N_ROWS-1)  ]

def motion_kinematics(points):
    for i, p in enumerate(points):
        if p.pinned: continue
        # print(p.pos, p.old_pos)
        vel = p.pos - p.old_pos
        p.old_pos = p.pos.copy()
        p.pos += vel * FRICTION
        p.pos += GRAVITY
        if i == 0:  # pull the bottom left point away to check that side connections exist
            p.pos += np.array([-0.7, 0.0])

def motion_rigidsticks(points):
    for stick in sticks:
        if stick.p_a.pinned and stick.p_b.pinned: continue
        # just straight up moves points towards/away from each other
        # as per https://www.youtube.com/watch?v=pBMivz4rIJY
        point_offset = stick.p_b.pos - stick.p_a.pos
        point_distance = np.linalg.norm(point_offset)
        offset = point_offset * (stick.len - point_distance)/point_distance

        # print(point_offset, point_distance, offset)

        if stick.p_a.pinned:
            stick.p_b.pos += offset
        elif stick.p_b.pinned:
            stick.p_a.pos -= offset
        else:
            stick.p_a.pos -= offset/2
            stick.p_b.pos += offset/2

# def motion_pinned(points):
#     for p in pinned_points:
#         p.pos = p.old_pos.copy()

plot_x, plot_y = [], []
def animate(i):
    motion_kinematics(points)
    # resolution handlers
    for _ in range(NUM_RESOLVE_STEPS):
        motion_rigidsticks(points)
    # motion_pinned(points)
    global plot_x, plot_y
    plot_x = [p.pos[0] for p in points]
    plot_y = [p.pos[1] for p in points]
    sc.set_offsets(np.c_[plot_x, plot_y])
    # input()
    # print(i, plot_x, plot_y)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    sc = ax.scatter(plot_x, plot_y)
    plt.xlim(-15, 5)
    plt.ylim(-15, 5)

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                frames=60, interval=100, repeat=True)
    plt.show()
#
# def render_points(points):
#     hl.set_aa()
#     # hl.set_xdata([p.pos[0] for p in points])
#     # hl.set_ydata([p.pos[1] for p in points])
#     plt.draw()
#
# if __name__ == '__main__':
#
#
#     for step in range(int(1e5)):
#         update_points(points)
#         render_points(points)



# Define the 8 vertices of the cube
vertices = np.array([\
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1]])
# Define the 12 triangles composing the cube
faces = np.array([\
    [0,3,1],
    [1,3,2],
    [0,4,7],
    [0,7,3],
    [4,5,6],
    [4,6,7],
    [5,1,2],
    [5,2,6],
    [2,3,6],
    [3,7,6],
    [0,1,5],
    [0,5,4]])

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "cube.stl"
cube.save('cube.stl')

