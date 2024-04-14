from mpi4py import MPI
from camera import raw_to_clip, clip_to_canvas
from frames import frame_to_camera



def mpi_render_wrap(pos, hsml, qty, frames, render_func, 
                    plot_n_save=None, tmp_path="../tmp/", map_prefix="map", img_prefix="img",
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
        print("No plot_n_save function is specified. No image will be produced.")
    
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
        w, h, p = raw_to_clip(cam, rho, hsml, pos, npix_x, npix_y)
        hi, pi = clip_to_canvas(h, p, npix_x, npix_y)
        
        grid = np.zeros([npix_x, npix_y])
        render_func(w, hi, pi, grid)
        comm.Barrier()
        grid_tot = comm.reduce(grid, MPI.SUM, 0)
        if rank == 0:
            # save data map
            map_fn = f"{tmp_path}/{map_prefix}_{str(frame_id).zfill(4)}.npy"
            np.save(f"{tmp_path}/{map_prefix}_{str(frame_id).zfill(4)}.npy", grid_tot)
            
            # save image
            img_fn = f"{tmp_path}/{img_prefix}_{str(frame_id).zfill(4)}"
            if plot_n_save is not None:
                plot_n_save(grid_tot, image_fn, tmp_path)