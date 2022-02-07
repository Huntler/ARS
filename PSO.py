from ast import Lambda
import random
from typing import List
import numpy as np
import time
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


INTEGER_MAX = np.iinfo(np.int(10)).max


class Particle:
    def __init__(self, pos_x, pos_y, vel_x, vel_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.p_best_x = pos_x
        self.p_best_y = pos_y
        self.vel_x = vel_x
        self.vel_y = vel_y

        self.bench_best = -1


class Swarm:
    def __init__(self, particles_num: int, params: List[float], x_range: List, y_range: List, max_steps, benchmark_func: Lambda, axis) -> None:
        self._benchmark_func = benchmark_func
        self._axis = axis

        self._a = params[0]
        self._b = params[1]
        self._c = params[2]

        self._history = []
        self._max_frames = max_steps
        self._time = 0
        self._cmap = get_cmap(particles_num+1)

        # best group position
        self._group_best_x = np.random.rand()
        self._group_best_y = np.random.rand()
        self._best_benchmark_val_group = -1

        self._x_min, self._x_max = x_range[0], x_range[1]
        self._y_min, self._y_max = y_range[0], y_range[1]

        # randomly initialize position and velocity of particles
        self._swarm = [Particle(random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]),
                                random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]))
                       for x in range(particles_num)]

    def update(self, frame_number, lr=0.01) -> None:
        # pause the animation max frames have been reached
        if frame_number >= self._max_frames:
            self._plot()
            return

        # do some special things on the initial frame
        if frame_number == 0:
            _x = [p.pos_x for p in self._swarm]
            _y = [p.pos_y for p in self._swarm]

            h = []
            for i, (__x, __y) in enumerate(zip(_x, _y)):
                __z = self._benchmark_func(__x, __y)
                h.append((__x, __y, __z))

            self._history.append(h)
            self._plot()
            return

        h = []
        # perform PSO step
        for i, particle in enumerate(self._swarm):
            # update personal particle high score
            benchmark_value = self._benchmark_func(particle.pos_x, particle.pos_y)
            if benchmark_value <= particle.bench_best or particle.bench_best == -1:
                particle.bench_best = benchmark_value
                particle.p_best_x = particle.pos_x
                particle.p_best_y = particle.pos_y

            # update the group's high score
            benchmark_value = self._benchmark_func(self._group_best_x, self._group_best_y)
            if self._best_benchmark_val_group == -1:
                benchmark_value = self._benchmark_func(particle.pos_x, particle.pos_y)
                self._group_best_x = particle.pos_x
                self._group_best_y = particle.pos_y

            elif benchmark_value <= self._best_benchmark_val_group:
                self._best_benchmark_val_group = benchmark_value
                self._group_best_x = particle.pos_x
                self._group_best_y = particle.pos_y

        for particle in self._swarm:
            # update velocity
            r_1 = np.random.rand()
            r_2 = np.random.rand()
            new_vel_x = self._a * particle.vel_x + self._b * r_1 * (
                        particle.p_best_x - particle.pos_x) + self._c * r_2 * (
                                self._group_best_x - particle.pos_x)
            new_vel_y = self._a * particle.vel_y + self._b * r_1 * (
                        particle.p_best_y - particle.pos_y) + self._c * r_2 * (
                                self._group_best_y - particle.pos_y)

            particle.vel_x = new_vel_x
            particle.vel_y = new_vel_y

            # and set new position
            particle.pos_x = particle.pos_x + particle.vel_x
            particle.pos_y = particle.pos_y + particle.vel_y

            # add to the history
            z = self._benchmark_func(particle.pos_x, particle.pos_y)
            h.append((particle.pos_x, particle.pos_y, z))
        self._history.append(h)

        # plot the particles
        self._plot()

        # randomly tune the 'a'
        if self._a >= 0.4:
            self._a -= lr

        # wait some time
        if self._time != 0:
            time.sleep(self._time)
    
    def _plot(self):
        for i in range(len(self._history)-1, 0, -1):
            for j, part in enumerate(self._history[i]):
                x, y, z = part
                if not (x < self._x_min or x > self._x_max):
                    if not (y < self._y_min or y > self._y_max):
                        a = (0.6**(len(self._history) - i))
                        self._axis.scatter3D(x, y, z, c=self._cmap(j), alpha=a, zorder=2)
        
        if len(self._history) == 10:
            self._history.pop(0)


def PSO(a, b, c, step_max, particles_num, benchmark_func, axis):
    """
    performs PSO
    :param benchmark_func: loss function
    :param a: learning constant
    :param b: learning constant
    :param c: learning constant
    :param step_max: PSO stops after step_max steps
    :param particles_num: number of partivles
    :return: None
    """

    # step counter
    k = 1

    # best group position
    group_best_x = -1
    group_best_y = -1
    best_benchmark_val_group = -1

    # randomly initialize position and velocity of particles
    rand_range = 15
    swarm = [Particle(random.randrange(-rand_range, rand_range), random.randrange(-rand_range, rand_range),
                      random.randrange(-rand_range, rand_range), random.randrange(-rand_range, rand_range))
             for x in range(particles_num)]

    # initial step
    for particle in swarm:
        benchmark_value = benchmark_func(particle.pos_x, particle.pos_y)
        particle.bench_best = benchmark_value
        particle.p_best_x = particle.pos_x
        particle.p_best_y = particle.pos_y
        if group_best_x is None or group_best_y is None:
            best_benchmark_val_group = benchmark_value
            group_best_x = particle.pos_x
            group_best_y = particle.pos_y
        else:
            group_bench_value = benchmark_func(group_best_x, group_best_y)
            if group_bench_value <= best_benchmark_val_group:
                best_benchmark_val_group = group_bench_value
                group_best_x = particle.pos_x
                group_best_y = particle.pos_y

        # update position and velocity
        r_1 = np.random.rand()
        r_2 = np.random.rand()
        new_vel_x = a * particle.vel_x + b * r_1 * (particle.p_best_x - particle.pos_x) + c * r_2 * (
            group_best_x - particle.pos_x)
        new_vel_y = a * particle.vel_y + b * r_1 * (particle.p_best_y - particle.pos_y) + c * r_2 * (
            group_best_y - particle.pos_y)

        particle.vel_x = new_vel_x
        particle.vel_y = new_vel_y

        particle.pos_x = particle.pos_x + particle.vel_x
        particle.pos_y = particle.pos_y + particle.vel_y

    k += 1

    # PSO loop
    while k <= step_max:
        x_positions = []
        y_positions = []

        # perform PSO step
        for particle in swarm:
            benchmark_value = benchmark_func(particle.pos_x, particle.pos_y)
            if benchmark_value <= particle.bench_best:
                particle.bench_best = benchmark_value
                particle.p_best_x = particle.pos_x
                particle.p_best_y = particle.pos_y
            benchmark_value = benchmark_func(group_best_x, group_best_y)
            if benchmark_value <= best_benchmark_val_group:
                best_benchmark_val_group = benchmark_value
                group_best_x = particle.pos_x
                group_best_y = particle.pos_y

                # update position and velocity
                r_1 = np.random.rand()
                r_2 = np.random.rand()
                new_vel_x = a * particle.vel_x + b * r_1 * (particle.p_best_x - particle.pos_x) + c * r_2 * (
                    group_best_x - particle.pos_x)
                new_vel_y = a * particle.vel_y + b * r_1 * (particle.p_best_y - particle.pos_y) + c * r_2 * (
                    group_best_y - particle.pos_y)

                particle.vel_x = new_vel_x
                particle.vel_y = new_vel_y

                particle.pos_x = particle.pos_x + particle.vel_x
                particle.pos_y = particle.pos_y + particle.vel_y

            x_positions.append(particle.pos_x)
            y_positions.append(particle.pos_y)

        axis.scatter(x_positions, y_positions)

        k += 1
