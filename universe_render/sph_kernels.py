import numpy as np
import math
from numba import jit

########################
##Homebrew SPH Kernels##
########################
# Note: 1. please keep the kernel simple
#       2. use @jit(nopython=True) to enable numba compilation
#       3. numba only supports a limited number of math functions


@jit(nopython=True)
def cubic_spline_2D(r, h):
    """
    2D cubic spline kernel. Taken from https://pysph.readthedocs.io/en/latest/reference/kernels.html
    """
    q = r / h
    fac = 10. * np.pi / 7.0 / h / h
    if q > 2.0:
        val = 0.
    elif q > 1.0:
        val = 0.25 * (2. - q) * (2. - q) * (2. - q)
    else:
        val = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return val * fac
