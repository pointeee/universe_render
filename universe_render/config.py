from sph_kernels import cubic_spline_2D

sph_kernel = cubic_spline_2D
npix_x, npix_y = 1920, 1080

data_path = "../../AGNcomp_/data/simba_50_070_gasdata.npy"
frames_path = "../frames.txt"
tmp_path  = "../tmp/"

SAVE_RENDER_ARR = True
SAVE_PLOT_IMAGE = True
