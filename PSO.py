import random
import numpy


class Particle:
    def __init__(self, pos_x, pos_y, vel_x, vel_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.p_best_x = pos_x
        self.p_best_y = pos_y
        self.vel_x = vel_x
        self.vel_y = vel_y

        self.bench_best = None


def PSO(a, b, c, step_max, particles_num, benchmark_func):
    """
    performs PSO
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
    group_best_x = None
    group_best_y = None
    best_benchmark_val_group = None

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
        r_1 = numpy.random.seed(0)
        r_2 = numpy.random.seed(42)
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
                r_1 = numpy.random.seed(0)
                r_2 = numpy.random.seed(42)
                new_vel_x = a * particle.vel_x + b * r_1 * (particle.p_best_x - particle.pos_x) + c * r_2 * (
                        group_best_x - particle.pos_x)
                new_vel_y = a * particle.vel_y + b * r_1 * (particle.p_best_y - particle.pos_y) + c * r_2 * (
                        group_best_y - particle.pos_y)

                particle.vel_x = new_vel_x
                particle.vel_y = new_vel_y

                particle.pos_x = particle.pos_x + particle.vel_x
                particle.pos_y = particle.pos_y + particle.vel_y

        k += 1
