import numpy as np

def from_npy(filename):
    data = np.load(filename).T
    n_part = data.shape[0]
    hsml = data[:,5]
    rho = data[:,4]
    temp = data[:,3]
    pos = data[:,0:3]

    return pos, hsml, rho, temp