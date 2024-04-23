from universe_render.camera import Camera, raw_to_clip, clip_to_canvas
from universe_render.snap_io import from_npy
from universe_render.render import render_func_factory
from universe_render.plot import rho_map # redua
from universe_render.sph_kernels import kernels
from universe_render.frames import hash_file, read_frame_file, frame_to_camera
from universe_render.mpi_wrapper import mpi_render_wrap

import numpy as np
import configparser
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

config = configparser.ConfigParser()
config.read('parallel_example_config.cfg')
npix_x = int(config["setting"]["npix_x"])
npix_y = int(config["setting"]["npix_y"])

data_path = config["paths"]["data_path"]
frames_path = config["paths"]["frames_path"]
tmp_path = config["paths"]["tmp_path"]
file_prefix = config["paths"]["file_prefix"]

hinv = config["processing"]["hinv"] == "True"
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
    
    mpi_render_wrap(pos, hsml, rho, frames, render_cpu, npix_x, npix_y, plot_n_save=rho_map)
    
    