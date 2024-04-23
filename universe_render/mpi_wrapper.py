from mpi4py import MPI
import numpy as np
import sys

from .camera import raw_to_clip, clip_to_canvas
from .frames import frame_to_camera

import warnings


def mpi_render_wrap(pos, hsml, qty, frames, render_func, npix_x, npix_y, use_hinv=False,
                    plot_n_save=None, tmp_path="../tmp/", movie_path="../movies/",map_prefix="map", img_prefix="img",
                    MPI=MPI):
    """
    An MPI wrapper for the render task
    ------
    pos    - ndarray([n_part, 3]) 
             the position of SPH particles in the scene.
    hsml   - ndarray([n_part,])
             the "size" of SPH particles in the scene.
    qty    - ndarray([n_part,])
             the quantity to render.
    frames - a list of frames that can be parsed with frames.frame_to_camera
             determine the camera info.
    render_func - the render function produced by render_func_factory
    plot_n_save - a function to save intermediate images.
                  If None then no image will be produced.
    tmp_path    - The path to save intermediate results.
    map_prefix  - The prefix for saved map files
    img_prefix  - The prefix for saved img files
    MPI         - The MPI api to use. By default mpi4py.MPI
    """
    # MPI init
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    # Check plot&save function
    if (plot_n_save is None) and (rank==0):
        warnings.warn("No plot_n_save function is specified. No image will be produced.")
       
    # Check hinv option #FIXME we can assign more info on the function
    if use_hinv and (rank==0):
        warnings.warn("Using hinv option. Please ensure the SPH kernel take hinv as input.")
        
    sys.stdout.flush()
    
    # Task decomposition 
    npart = len(hsml)
    part_per_thread = int(npart // size + 1)
    ip_start, ip_end = part_per_thread*(rank+0), min(part_per_thread*(rank+1), npart)
    
    # Data decomposition
    pos  = pos[ip_start:ip_end]
    hsml = hsml[ip_start:ip_end]
    qty  = qty[ip_start:ip_end]
    
    # Render loop
    for frame_id, frame in enumerate(frames):
        cam = frame_to_camera(frame)
        w, h, p = raw_to_clip(cam, qty, hsml, pos, npix_x, npix_y)
        hi, pi = clip_to_canvas(h, p, npix_x, npix_y)
        
        grid = np.zeros([npix_x, npix_y], dtype=float)
        if use_hinv:
            hi_inv = 1. / hi
            render_func(w, hi, hi_inv, pi, grid)
        else:
            render_func(w, hi, pi, grid)
        comm.Barrier()
        grid_tot = comm.reduce(grid, MPI.SUM, 0)
        comm.Barrier()
        if rank == 0:
            # save data map
            map_fn = f"{tmp_path}/{map_prefix}_{str(frame_id).zfill(4)}.npy"
            np.save(f"{tmp_path}/{map_prefix}_{str(frame_id).zfill(4)}.npy", grid_tot)
            
            # save image
            img_fn = f"{tmp_path}/{img_prefix}_{str(frame_id).zfill(4)}.jpg"
            if plot_n_save is not None:
                plot_n_save(grid_tot, img_fn, tmp_path)
                
    if rank == 0:
        print("Render finished! Please generate the movie with")
        print(f"    ffmpeg -framerate 24 -i {tmp_path}{img_prefix}_%04d.jpg -c:v libx264 -profile:v high -pix_fmt yuv420p {movie_path}output.mp4")