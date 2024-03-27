from collections import namedtuple
import numpy as np

Point = namedtuple("Point", "x y")


def check_collision(point_i: Point, point_j: Point, ri, rj) -> bool:
    return ri + rj >= np.sum(np.power([point_i.x - point_j.x, point_i.y - point_j.y], 2))


def collision_point(pi: Point, pj: Point, ri, rj) -> Point | None:
    if check_collision(pi, pj, ri, rj):
        den = ri + rj
        cx = (pi.x * rj + pj.x * ri) / den
        cy = (pi.y * rj + pj.y * ri) / den
        return Point(cx, cy)
    return None
