import numpy as np
import matplotlib.pyplot as plt

from config import tmp_path

def rho_map(arr, fn, ftype="jpg"):
    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(np.log10(arr).T, cmap="cubehelix", origin='lower')
    plt.clim(-9, -4)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{tmp_path}/{fn}.{ftype}")
    plt.close()