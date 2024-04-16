import math
import numpy as np 
import numba 
from numba import njit

@njit
def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def render_func_factory(sph_kernel_hinv, npix_x, npix_y, use_hinv=False):
    if use_hinv:
    def render_cpu(w, h, h_inv, p, grid):
        nx, ny = npix_x, npix_y
        npart = w.size
        for ip in range(npart):
            ix_start, ix_end = int(max(0, p[ip,0] - 2*h[ip])), int(min(nx, p[ip,0] + 2*h[ip]))
            iy_start, iy_end = int(max(0, p[ip,1] - 2*h[ip])), int(min(ny, p[ip,1] + 2*h[ip]))
            for ix in range(ix_start, ix_end):
                for iy in range(iy_start, iy_end):
                    r = dist(p[ip,0], p[ip,1], ix, iy)
                    grid[ix, iy] = sph_kernel_hinv(r, hinv[ip])*w[ip] + grid[ix, iy]
        return grid
    else:
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
    return render_cpu

