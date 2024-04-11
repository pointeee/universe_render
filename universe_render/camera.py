import numpy as np
from numba import jit

class Camera(object):
    def __init__(self, pos, taD, up=np.array([0, 0, 1]), fov=90, zNear=0.1, zFar=5000):
        """
        Create a "Camera" object. The implementation follows the notation in OpenGL.
        See https://learnopengl.com/Getting-started/Camera for a visualization.
        pos:   np.array([3, ])
               The position of camera in the coordinate space.
        taD:   np.array([3, ])
               The direction of camera pointing. Note this is different from OpenGL, 
               where this "direction vector" points the opposite direction
        up:    np.array([3, ])
               The "up" axis. help to set the camera coordinate system.
        fov:   float
               field-of-view in degree. This FOV is along the horizontal direction
        zNear: float
               the z-axis truncation at the near end.
        zFar:  float 
               the z-axis truncation at the far end.
        """
        self.pos = pos
        self.taD_ = taD # this is the direction the camera pointing
        self.taD = self.norm(-taD) # this is the direction of +z in camera coordi system
        self.upV = up
        self.riD = self.norm(np.cross(self.upV, self.taD))
        self.upD = self.norm(np.cross(self.taD, self.riD))

        self.fov   = fov / 180 * np.pi
        self.zNear = zNear
        self.zFar  = zFar

    def look_at_matrix(self): # aka view matrix
        """
        Return the look at matrix (aka view matrix) of the camera object.
        """
        A = np.array([[self.riD[0], self.riD[1], self.riD[2], 0          ], 
                       [self.upD[0], self.upD[1], self.upD[2], 0          ],
                       [self.taD[0], self.taD[1], self.taD[2], 0          ],
                       [0          , 0          , 0          , 1          ]])
        B = np.array([[1          , 0          , 0          ,-self.pos[0]], 
                       [0          , 1          , 0          ,-self.pos[1]],
                       [0          , 0          , 1          ,-self.pos[2]],
                       [0          , 0          , 0          , 1          ]])
        return A@B

    @classmethod
    def project_matrix(cls, l, r, b, t, n, f):
        """
        Return the project matrix for any input l, r, b, t, n, f.

        l - left (<0)
        r - right (>0)
        t - top (>0)
        b - bottom (<0)
        n - near
        f - far
        """
        M = np.array([[2*n/(r-l), 0        , (r+l)/(r-l),            0],
                       [0        , 2*n/(t-b), (t+b)/(t-b),            0],
                       [0        , 0        ,-(f+n)/(f-n), -2*f*n/(f-n)],
                       [0        , 0        ,          -1           , 0]])
        return M

    @classmethod
    def project_matrix_sym(cls, r, t, n, f):
        """
        Return the project matrix for any input r, t, n, f.
        Assuming l = -r, b = -t.

        r - right (>0)
        b - bottom (<0)
        n - near
        f - far
        """
        return cls.project_matrix(-r, r, -t, t, n, f)

    def project_matrix_from_fov(self):
        """
        Return the project matrix based on the camera's FOV. 
        """
        r1 = np.tan(self.fov / 2)
        t1 = np.tan(self.fov / 2)
        return self.project_matrix_sym(r1*self.zNear, t1*self.zNear, self.zNear, self.zFar)

    def to_clip(self, h, pos_4):
        """
        Convert the particle 4-position & hsml in the scene into the clip space.
        h:     np.array([n_part, ])
               the hsml of particles in the coordinate space.
        pos_4: np.array([4, n_part])
               the 4-position of particles in the coordinate space.
               For a given 3-pos (x, y, z), its corresponding 4-pos is (x, y, z, 1).
        """
        view_pos_4 = self.look_at_matrix() @ pos_4
        clip_pos_4 = self.project_matrix_from_fov() @ view_pos_4
        w = clip_pos_4[3]
        clip_h     = h / np.abs(view_pos_4[2]) / np.tan(self.fov/2)
        clip_pos_4 = clip_pos_4 / w
        return clip_h, clip_pos_4

    def to_mask_clip(self, weight, h, pos_4, clip_x=1, clip_y=1):
        """
        Convert the particle 4-position & hsml in the scene into the clip space by calling self.to_mask.
        AND apply a mask to select the particels for render.
        
        weight: np.array([n_part, ])
                the weight array for each particle. This can be density or temperature for example.
        h:      np.array([n_part, ])
                the hsml of particles in the coordinate space.
        pos_4:  np.array([4, n_part])
                the 4-position of particles in the coordinate space.
                For a given 3-pos (x, y, z), its corresponding 4-pos is (x, y, z, 1).
        clip_x: float
        clip_y: float
                determine the x-y region for render as following:
                [(-clip_x, clip_y) --- ( clip_x, clip_y)]
                [                                       ]
                [                                       ]
                [(-clip_x,-clip_y) --- ( clip_x,-clip_y)]
        """
        clip_h, clip_coord = self.to_clip(h, pos_4)
        clip_h = np.array(clip_h)
        mask = (clip_coord[0]>-clip_x) * (clip_coord[0]<clip_x) * (clip_coord[1]>-clip_y) * (clip_coord[1]<clip_y) * (clip_coord[2]>-1) * (clip_coord[2]<1)
        masked_w = weight[mask]
        masked_h = clip_h[mask]
        masked_coord = clip_coord[0:2,mask].T
        return masked_w, masked_h, masked_coord, mask
    
    @classmethod
    def norm(cls, vec):
        return vec / np.sum(vec**2.)**.5

def raw_to_clip(c, weight, hsml, pos, npix_x, npix_y):
    """
    a wrapper to convert the particle data to masked clip space.
    ------
    c - camera.Camera object
    pos  - ndarray([n_part, 3]) position of particles
    hsml - ndarray([n_part]) "size" of particles
    weight - ndarray([n_part]) the quantity to map on the canvas
    
    npix_x - int. number of pixels along the x axis
    npix_y - int. number of pixels along the y axis
    ------
    w - ndarray([n_part_masked]) the quantity to map on the clip space after the mask
    h - ndarray([n_part_masked]) the size of particles on the clip space after the mask
    p - ndarray([n_part_masked, 2]) the position of particels on the clip space after the mask
    """
    n_part = len(hsml)
    pos_4 = np.hstack([pos, np.ones(shape=(n_part, 1))]).T
    w, h, p, _ = c.to_mask_clip(weight, hsml, pos_4, clip_x=npix_x/npix_y, clip_y=1)
    return w, h, p
    
def clip_to_canvas(h, p, npix_x, npix_y):
    """
    a warpper to conver the particles from the clip space to the canvas
    [(-nx/ny, 1) --- ( nx/ny, 1)]       [( 0,ny) --- (nx,ny)]
    [                           ]  ---> [                   ]
    [                           ]  ---> [                   ]
    [(-nx/ny,-1) --- ( nx/ny,-1)]       [( 0, 0) --- (nx, 0)]
    ------
    h - ndarray([n_part_masked])
    p - ndarray([n_part_masked, 2])
    ------
    h_index - ndarray([n_part_masked])
    p_index - ndarray([n_part_masked, 2])
    """
    h_index = h * npix_y / 2
    p_index = p * npix_y / 2 + np.array([npix_x/2, npix_y/2])
    return h_index, p_index
    
    