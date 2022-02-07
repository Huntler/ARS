from tkinter import SW
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from benchmark_functions import rosenbrock, rastrigin

from PSO import Swarm

# Fixing random state for reproducibility
np.random.seed(19680801)

# set initial limits
x_min, x_max = -2.5, 2.5
y_min, y_max = -1, 3
z_min, z_max = 0, 2500

# create ranges of x and y
x = np.arange(x_min, x_max, 0.1)
y = np.arange(y_min, y_max, 0.1)
x, y = np.meshgrid(x, y)

# load the landscape
# FIXME: use argv to load the wanted landscape
z = rosenbrock(x, y, a=0, b=50)

# Attaching 3D axis to the figure
fig = plt.figure(figsize=(8, 8))
# FIXME: change title if different function is used
fig.suptitle("Rastrigin Benchmark")

# plot the 2d plane
# ax2d = fig.add_subplot(1, 2, 1)
# ax2d.imshow(z, cmap=cm.coolwarm, origin='lower')

# plot the 3d plane
ax3d = fig.add_subplot(1, 1, 1, projection='3d')
ax3d.set_xlim3d(x_min, x_max)
ax3d.set_ylim3d(y_min, y_max)
ax3d.set_zlim3d(z_min, z_max)
ax3d.view_init(elev=90, azim=0)

# plot the fitness function
surf = ax3d.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.3, linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

# initialize the swarm
max_iterations = 1000
swarm = Swarm(20, params=[0.9, 2, 2], x_range=[x_min, x_max], y_range=[
              y_min, y_max], benchmark_func=rosenbrock, max_steps=max_iterations, axis=ax3d)

def update(frame_number):
    ax3d.clear()
    swarm.update(frame_number)
    ax3d.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.3, linewidth=0, antialiased=False)

# show the animation and run the simulation using swarm.update
animation = FuncAnimation(fig, update, interval=1)
plt.show()
