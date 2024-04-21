from universe_render.camera import Camera, raw_to_clip, clip_to_canvas
from universe_render.snap_io import from_npy
from universe_render.render import render_func_factory
from universe_render.plot import rho_map # redua
from universe_render.sph_kernels import kernels
from universe_render.frames import hash_file, read_frame_file, frame_to_camera

import numpy as np
import configparser
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

config = configparser.ConfigParser()
config.read('Config.cfg')
npix_x = int(config["setting"]["npix_x"])
npix_y = int(config["setting"]["npix_y"])

data_path = config["path"]["data_path"]
frames_path = config["path"]["frames_path"]
tmp_path = config["path"]["tmp_path"]
file_prefix = config["path"]["file_prefix"]

hinv=bool(config["processing"]["hinv"])
if hinv:
    sph_kernel = kernels[config["setting"]["sph_kernel"]+"_hinv"]
else:
    sph_kernel = kernels[config["setting"]["sph_kernel"]]

render_cpu = render_func_factory(sph_kernel, npix_x, npix_y)

if __name__ == "__main__":
    # read frame info
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
            np.save(f"{tmp_path}/{file_prefix}_image_{str(frame_id).zfill(4)}.npy", grid_tot)
            image_fn = f"{file_prefix}_image_{str(frame_id).zfill(4)}"
            rho_map(grid_tot, image_fn, tmp_path)
            
    
