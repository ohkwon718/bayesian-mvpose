
import numpy as np
import itertools
from operator import itemgetter
from scipy.linalg import lstsq


def get_lookat_vec_in_global(camera_matrixs, size_hw):
    p0 = np.array(size_hw)[::-1]/2
    R = camera_matrixs[...,:3]
    A = R - np.append(p0,0)[None,:,None] @ R[:,2:3,:]
    invA = np.linalg.inv(A)
    z = (invA @ np.array([0,0,1])[None,:,None])[...,0]
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return z


def is_front_of_view(xyz, camera_matrixs):
    """
    param xyz : np.array{ n x 3 }
    param camera_matrixs : np.array{ ... 3 x 4 }
    """
    
    if len(xyz.shape) == 1:
        xyz = xyz.reshape(1,-1)
    xyz1 = np.ones((len(xyz),4))
    xyz1[:,:3] = xyz
    xy1 = np.swapaxes(camera_matrixs @ xyz1.T, -1, -2)
    
    return xy1[...,2] > 0