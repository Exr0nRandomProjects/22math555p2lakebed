import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import matplotlib.animation



from dataclasses import dataclass

GRAVITY = np.array([0, 0, -0.1])
FRICTION = 0.9

NUM_RESOLVE_STEPS = 8

N_COLS = 30
N_ROWS = 30

SIZE = 100

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
    Point(pos=np.array([x, y, 0]), old_pos=np.array([x, y, 0]), pinned=False)
    for x in np.linspace(0, SIZE, N_COLS) for y in np.linspace(0, SIZE, N_ROWS) ]

for i, p in enumerate(points):
    # if i % N_ROWS == N_ROWS-1 and np.random.random() < 0.3:
    if p.pos[0] == 0 or p.pos[0] == SIZE or p.pos[1] == 0 or p.pos[1] == SIZE:
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
        # if i == 0:  # pull the bottom left point away to check that side connections exist
        #     p.pos += np.array([-0.7, 0.0])

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

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
sc = ax.scatter([], [], [])
line_ax = fig.add_subplot(122)
max_vel_data = []
avg_vel_data = []
max_vel_line, = line_ax.plot([], [], label="maximum velocity")
avg_vel_line, = line_ax.plot([], [], label="average velocity")
line_ax.set_xlabel('iterations')
line_ax.legend()

def animate(i):
    motion_kinematics(points)
    # resolution handlers
    for _ in range(NUM_RESOLVE_STEPS):
        motion_rigidsticks(points)


    plot_x = [p.pos[0] for p in points]
    plot_y = [p.pos[1] for p in points]
    plot_z = [p.pos[2] for p in points]
    # sc.set_offsets(np.c_[plot_x, plot_y, plot_z])
    sc._offsets3d = (plot_x, plot_y, plot_z) # https://stackoverflow.com/a/41609238

    # tracking line chart
    vels = [np.linalg.norm(p.pos - p.old_pos) for p in points]
    max_vel_data.append(max(vels))
    avg_vel_data.append(sum(vels)/len(vels))
    max_vel_line.set_ydata(max_vel_data)
    max_vel_line.set_xdata(range(len(max_vel_data)))
    avg_vel_line.set_ydata(avg_vel_data)
    avg_vel_line.set_xdata(range(len(avg_vel_data)))
    line_ax.set_xlim(0, len(avg_vel_data))
    y_scale = max(max_vel_data)
    line_ax.set_ylim(-0.2*y_scale, 1.2*y_scale)

if __name__ == '__main__':


    # plot_x = [p.pos[0] for p in points]
    # plot_y = [p.pos[1] for p in points]
    # plot_z = [p.pos[2] for p in points]

    ax.set_xlim3d([-SIZE*0.2, SIZE*1.2])
    ax.set_ylim3d([-SIZE*0.2, SIZE*1.2])
    ax.set_zlim3d([-30, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    line_ax.set_ylim(0, 3)
    # line_ax.set_yscale('log')

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                frames=60, interval=100, blit=False)


    plt.show()

    print("shape confirmed. exporting to stl...")


# # Define the 8 vertices of the cube
# vertices = np.array([\
#     [-1, -1, -1],
#     [+1, -1, -1],
#     [+1, +1, -1],
#     [-1, +1, -1],
#     [-1, -1, +1],
#     [+1, -1, +1],
#     [+1, +1, +1],
#     [-1, +1, +1]])
# # Define the 12 triangles composing the cube
# faces = np.array([\
#     [0,3,1],
#     [1,3,2],
#     [0,4,7],
#     [0,7,3],
#     [4,5,6],
#     [4,6,7],
#     [5,1,2],
#     [5,2,6],
#     [2,3,6],
#     [3,7,6],
#     [0,1,5],
#     [0,5,4]])
#
# # Create the mesh
# cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces):
#     for j in range(3):
#         cube.vectors[i][j] = vertices[f[j],:]
#
# # Write the mesh to file "cube.stl"
# cube.save('cube.stl')
#
