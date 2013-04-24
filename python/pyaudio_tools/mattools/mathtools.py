__author__ = 'Adam Miller'
import math


def sign(x):
    if x < 0:
        return -1
    return 1


def to_radians(x):
    """
    Convert from degrees to radians
    """
    return float(x) / 180 * math.pi


def to_degrees(x):
    """
    Convert from radians to degrees
    """
    return float(x) / math.pi * 180
