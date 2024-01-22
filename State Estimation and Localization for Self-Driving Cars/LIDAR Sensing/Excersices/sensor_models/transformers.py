import numbers
from numpy import cos, sin, sqrt, arctan2, arcsin


def inverse_transformation(r: numbers.Real, alpha: numbers.Real, epsilon: numbers.Real) -> (float, float, float):
    """

    :param r:  range
    :param alpha: azimuth
    :param epsilon: elevation
    :return: x,y,z
    """
    return r * cos(alpha) * cos(epsilon), r * sin(alpha) * cos(epsilon), r * sin(epsilon)


def forward_transformation(x: numbers.Real, y: numbers.Real, z: numbers.Real) -> (float, float, float):
    """

    :param x:
    :param y:
    :param z:
    :return: r (range), alpha (azimut), epsilon (elevation)
    """
    magnitude = sqrt(x ** 2 + y ** 2 + z ** 2)
    return magnitude, arctan2(y, x), arcsin(z, magnitude)
