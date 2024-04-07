import numpy as np
import math
from numba import jit

@jit(nopython=True)
def cubic_spline_2D(r, h):
    q = r / h
    fac = 10. * np.pi / 7.0 / h / h
    if q > 2.0:
        val = 0.
    elif q > 1.0:
        val = 0.25 * (2. - q) * (2. - q) * (2. - q)
    else:
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac
