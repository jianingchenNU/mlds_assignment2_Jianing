import numpy as np


def piecewise(t, points):
    """Linear interpolation through (step, value) points, constant after the last one."""
    xs, ys = zip(*points)
    return float(np.interp(t, xs, ys))
