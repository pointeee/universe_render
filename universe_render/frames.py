import numpy as np
import quaternion
from scipy.interpolate import interp1d
import hashlib
from camera import Camera

############
## Frames ##
############
# Note: One frame is actually a numpy array, which contains 
#       all the parameter for creation of new camera object.
#       Frame  = (t, posx, posy, posz, dirx, diry, dirz, fov, n, f)
#       Frames = list of frames

# to determine the prefix in the saved files
def hash_file(filename):
    """"This function returns the SHA-1 hash
    of the file passed into it"""

    # make a hash object
    h = hashlib.sha256()

    # open file for reading in binary mode
    with open(filename,'rb') as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()

# convert frame into to a camera object
def frame_to_camera(frame):
    """
    Parse one frame and convert it into a camera object.
    """
    time = frame[0]
    posc = frame[1:4]
    dirc = frame[4:7]
    fovc = frame[7]
    neac = frame[8]
    farc = frame[9]
    return Camera(posc, dirc, fov=fovc, zNear=neac, zFar=farc)
    

# frame io
def read_frame_file(filename):
    """
    Read frames from a given file.
    """
    with open(filename) as f:
        lines = f.readlines()[1:]
    return np.array([[float(_) for _ in line.split()] for line in lines])

def write_frame_file(frames, filename):
    """
    Write frames into a given file.
    """
    with open(filename, "w") as f:
        f.write("# t, posx, posy, posz, dirx, diry, dirz, fov, n, f\n")
        for frame in frames:
            f.write("{:=+8E} {:=+8E} {:=+8E} {:=+8E} {:=+8E} {:=+8E} {:=+8E} {:=+8E} {:=+8E} {:=+8E}\n".format(*frame))

# frame interpolation
def keyframes_to_all_frames(kf, timestep=1):
    """
    Interpolate between the keyframes provided.
    """
    n_kf = kf.shape[0]
    time = kf[:,0]
    posc = kf[:,1:4]
    dirc = kf[:,4:7]
    fovc = kf[:,7:8]
    neac = kf[:,8:9]
    farc = kf[:,9:10]

    time_interp = np.arange(time[0], time[-1], timestep)
    posc_interp = interp1d(time, posc, kind='cubic', axis=0)(time_interp)
    dirc_interp = []
    for i in range(n_kf-1):
        v_s, v_e = dirc[i], dirc[i+1]
        nstep = int((time[i+1]-time[i]) / timestep)
        dirc_interp += interp_rot(v_s, v_e, nstep)
    dirc_interp = np.array(dirc_interp)
    fovc_interp = interp1d(time, fovc, kind='cubic', axis=0)(time_interp)
    neac_interp = interp1d(time, neac, kind='cubic', axis=0)(time_interp)
    farc_interp = interp1d(time, farc, kind='cubic', axis=0)(time_interp)
    
    time_interp = np.expand_dims(time_interp, 1)

    return np.hstack([time_interp, posc_interp, dirc_interp, fovc_interp, neac_interp, farc_interp])

# supporting functions
def get_rotation(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    ax = np.cross(v1, v2)
    ax = ax / np.linalg.norm(ax)

    theta = np.arccos(np.dot(v1, v2))
    return ax, theta

def construct_q_rot(ax, theta):
    return np.quaternion(np.cos(theta/2), ax[0]*np.sin(theta/2), ax[1]*np.sin(theta/2), ax[2]*np.sin(theta/2))

def rotate(v, q_rot):
    q_v  = np.quaternion(0, v[0], v[1], v[2])
    q_vv = q_rot * q_v * q_rot.conjugate()
    return np.array([q_vv.x, q_vv.y, q_vv.z])

def interp_rot(v_s, v_e, nstep):
    if (v_s == v_e).all():
        return [v_s, ] * nstep
    ax, theta = get_rotation(v_s, v_e)
    theta_step = theta / nstep
    q_rot_step = construct_q_rot(ax, theta_step)
    v_interp = v_s
    v_interp_list = []
    for _ in range(nstep):
        v_interp_list.append(v_interp)
        v_interp = rotate(v_interp, q_rot_step)
    return v_interp_list