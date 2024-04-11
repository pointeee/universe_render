from sph_kernels import cubic_spline_2D

##########################
## Config for rendering ##
##########################
# Note: this file should not depend on any code inside the repo except for sph_kernels

sph_kernel = cubic_spline_2D # Kernel used in rendering
npix_x, npix_y = 1920, 1080 # Number of pixels along xy axis

data_path = "../../AGNcomp_/data/simba_50_070_gasdata.npy" # the path to data file
frames_path = "../frames.txt" # the path to frame file
tmp_path  = "../tmp/" # the path to save intermediate results

SAVE_RENDER_ARR = True # Not Implemented Yet
SAVE_PLOT_IMAGE = True # Not Implemented Yet
