from collections import namedtuple

import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt

Point = namedtuple("Point", "x y")


def check_collision(point_i: Point, point_j: Point, ri, rj) -> bool:
    return ri + rj >= np.sqrt(np.sum(np.power([point_i.x - point_j.x, point_i.y - point_j.y], 2)))


def collision_point(pi: Point, pj: Point, ri, rj) -> Point | None:
    if check_collision(pi, pj, ri, rj):
        den = ri + rj
        cx = (pi.x * rj + pj.x * ri) / den
        cy = (pi.y * rj + pj.y * ri) / den
        return Point(cx, cy)
    return None


def set_plot_limits(car1: 'SimpleCar', car2: 'SimpleCar', ax_: matplotlib.axes.Axes):
    y_lim = car1.fr_distance() / 2 + 0.25

    x_max = max(car1.get_edges() + car2.get_edges()) + 0.5
    x_min = min(car1.get_edges() + car2.get_edges()) - 0.5

    ax_.set_ylim((-y_lim, y_lim))
    ax_.set_xlim((x_min, x_max))


class SimpleCar:
    def __init__(self, center: float, f_distance: float, r_distance: float = None):
        """

        :param center: Car center
        :param f_distance: Front bumper distance
        :param r_distance: Rear bumper distance
        """
        self.center = center
        if r_distance is None:
            r_distance = f_distance
        self.r_distance = r_distance
        self.f_distance = f_distance

        self.f_center = self.r_center = self.c_radius = self.r_radius = self.f_radius = 0

        self.movement(0, 0)

        self.c_radius = abs(self.f_center - self.r_center) / 2
        self.f_radius = abs(self.f_center - self.center)
        self.r_radius = abs(self.r_center - self.center)

    def movement(self, vel: float, dt: float):
        self.center += vel * dt
        self.f_center = self.center + self.f_distance / 2
        self.r_center = self.center - self.r_distance / 2

    def get_edges(self) -> tuple[float, float]:
        return self.r_center - self.r_distance, self.f_center + self.f_distance

    def fr_distance(self):
        return self.f_center - self.r_center

    def generate_circles(self, ax_: matplotlib.axes.Axes, plotter=plt):
        c_circle = plotter.Circle((self.center, 0), self.c_radius, fill=False)
        r_circle = plotter.Circle((self.r_center, 0), self.r_radius, fill=False)
        f_circle = plotter.Circle((self.f_center, 0), self.f_radius, fill=False)
        ax_.add_patch(r_circle)
        ax_.add_patch(c_circle)
        ax_.add_patch(f_circle)

    def get_circle_and_radius(self) -> list[tuple[Point, float]]:
        c_p = Point(self.center, 0), self.c_radius
        r_p = Point(self.r_center, 0), self.r_radius
        f_p = Point(self.f_center, 0), self.f_radius
        return [c_p, r_p, f_p]

    def detect_collision(self, other: 'SimpleCar') -> bool:
        for pi, ri in self.get_circle_and_radius():
            for pj, rj in other.get_circle_and_radius():
                if collision_point(pi, pj, ri, rj):
                    return True
        return False
