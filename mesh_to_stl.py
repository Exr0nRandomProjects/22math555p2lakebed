import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import matplotlib.animation



from dataclasses import dataclass

@dataclass
class Point:
    pos: np.ndarray
    old_pos: np.ndarray


points = [
    Point(pos=np.array([x, y]), old_pos=np.array([x+0.1, y+0.1]))
    for x in np.linspace(0, 10, 11) for y in np.linspace(0, 10, 11) ]

def update_points(points):
    for p in points:
        p.pos += 0.1



plot_x, plot_y = [], []
def animate(i):
    update_points(points)
    global plot_x, plot_y
    # plot_x.append(np.random.random()*10)
    # plot_y.append(np.random.random()*10)
    plot_x = [p.pos[0] for p in points]
    plot_y = [p.pos[1] for p in points]
    sc.set_offsets(np.c_[plot_x, plot_y])
    print(i, plot_x, plot_y)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    sc = ax.scatter(plot_x, plot_y)
    plt.xlim(-5,15)
    plt.ylim(-5,15)

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                frames=100, interval=100, repeat=True)
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

