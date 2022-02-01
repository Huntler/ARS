import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from benchmark_functions import rosenbrock, rastrigin

# Fixing random state for reproducibility
np.random.seed(19680801)

# set initial limits
x_min, x_max = -10, 10
y_min, y_max = -10, 10
z_min, z_max = 0, 1

# create ranges of x and y
x = np.arange(x_min, x_max, 0.01)
y = np.arange(y_min, y_max, 0.01)
x, y = np.meshgrid(x, y)

# load the landscape
# FIXME: use argv to load the wanted landscape
z = rastrigin(x, y)

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# plot the fitness function
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
