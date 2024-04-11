import math
import numpy as np 
import numba 
from numba import njit
# import numba_mpi

from config import sph_kernel, npix_x, npix_y


@njit
def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


@njit
def render_cpu(w, h, p, grid):
    nx, ny = npix_x, npix_y
    npart = w.size
    for ip in range(npart):
        ix_start, ix_end = int(max(0, p[ip,0] - 2*h[ip])), int(min(nx, p[ip,0] + 2*h[ip]))
        iy_start, iy_end = int(max(0, p[ip,1] - 2*h[ip])), int(min(ny, p[ip,1] + 2*h[ip]))
        for ix in range(ix_start, ix_end):
            for iy in range(iy_start, iy_end):
                r = dist(p[ip,0], p[ip,1], ix, iy)
                grid[ix, iy] = sph_kernel(r, h[ip])*w[ip] + grid[ix, iy]
    return grid