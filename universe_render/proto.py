from camera import Camera, raw_to_clip, clip_to_canvas
from snap_io import from_npy
from render import render_cpu
from config import npix_x, npix_y, data_path, frames_path
from plot import rho_map
from frames import hash_file, read_frame_file, frame_to_camera

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":
    # read frame info
    file_prefix = hash_file(frames_path)[0:8]
    frames = read_frame_file(frames_path)
    n_frame = frames.shape[0]
    
    # read particle data
    pos, hsml, rho, temp = from_npy(data_path)
    npart = len(hsml)
    
    # determine thread segment
    part_per_thread = int(npart // size + 1)
    ip_start, ip_end = part_per_thread*(rank+0), min(part_per_thread*(rank+1), npart)
    
    pos  = pos[ip_start:ip_end]
    hsml = hsml[ip_start:ip_end]
    rho  = rho[ip_start:ip_end]
    temp = temp[ip_start:ip_end]
    
    for frame_id, frame in enumerate(frames):
        cam = frame_to_camera(frame)
        w, h, p = raw_to_clip(cam, rho, hsml, pos, npix_x, npix_y)
        hi, pi = clip_to_canvas(h, p, npix_x, npix_y)
        
        grid = np.zeros([npix_x, npix_y])
        render_cpu(w, hi, pi, grid)
        comm.Barrier()
        grid_tot = comm.reduce(grid, MPI.SUM, 0)
        if rank == 0:
            np.save(f"../tmp/{file_prefix}_image_{str(frame_id).zfill(4)}.npy", grid_tot)
            image_fn = f"{file_prefix}_image_{str(frame_id).zfill(4)}"
            rho_map(grid_tot, image_fn)
