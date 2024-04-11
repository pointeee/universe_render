import numpy as np
import matplotlib.pyplot as plt

from config import tmp_path

####################
## Plot Functions ##
####################
# Note: PLZ define yours yourself

def rho_map(arr, fn, ftype="jpg"):
    dpi=240
    fig = plt.figure(figsize=(arr.shape[0]/dpi, arr.shape[1]/dpi), dpi=dpi)
    axes=fig.add_axes([0,0,1,1])    
    plt.imshow(np.log10(arr).T, cmap="cubehelix", origin='lower',clim=(-9, -4))
    axes.set_axis_off()
    plt.savefig(f"{tmp_path}/{fn}.{ftype}", bbox_inches='tight', pad_inches=0)
    plt.close()