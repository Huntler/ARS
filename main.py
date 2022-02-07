import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from benchmark_functions import rosenbrock, rastrigin

from PSO import PSO

PSO(1, 2, 1, 20, 20, rosenbrock)

# Fixing random state for reproducibility
np.random.seed(19680801)

# set initial limits
x_min, x_max = -2, 2
y_min, y_max = -0.5, 3
z_min, z_max = 0, 2500

# create ranges of x and y
x = np.arange(x_min, x_max, 0.01)
y = np.arange(y_min, y_max, 0.01)
x, y = np.meshgrid(x, y)

# load the landscape
# FIXME: use argv to load the wanted landscape
z = rosenbrock(x, y)

# Attaching 3D axis to the figure
fig = plt.figure(figsize=(8, 4))
# FIXME: change title if different function is used
fig.suptitle("Rastrigin Benchmark")

# plot the 2d plane
ax2d = fig.add_subplot(1, 2, 1)
ax2d.imshow(z, cmap=cm.coolwarm)

# plot the 3d plane
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.set_xlim3d(x_min, x_max)
ax3d.set_ylim3d(y_min, y_max)
ax3d.set_zlim3d(z_min, z_max)

# plot the fitness function
surf = ax3d.plot_surface(x, y, z, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

# Rotate the axes and update
for angle in range(0, 360, 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180
    ax3d.view_init(10, angle_norm)

    plt.title("3D Visualitzation")
    plt.draw()
    plt.pause(.01)
