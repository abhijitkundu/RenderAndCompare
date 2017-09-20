"""
Asserts quanitiies like viewpoint, bbx
"""

import numpy as np
from geometry import is_rotation_matrix

def assert_viewpoint(vp):
    """Asserts if vp is a valid viewpoint object"""
    assert isinstance(vp, np.ndarray), "viewpoint needs to be a numpy array"
    assert vp.shape == (3, ), "viewpoint needs to 3D vector but got shape {}".format(vp.shape)
    assert (vp >= -np.pi).all() and (vp < np.pi).all(), "Expects viewpoint to be in [-p1, pi), but got {}".format(vp)

def assert_bbx(bbx):
    """Assert if bbx is a valid bbx object"""
    assert isinstance(bbx, np.ndarray), "bbx needs to be a numpy array"
    assert bbx.shape == (4, ), "bbx needs to 4D vector but got shape {}".format(bbx.shape)
    assert np.all(bbx[:2] <= bbx[2:]), "Invalid bbx = {}".format(bbx)

def assert_coord2D(coord):
    """Assert if coord is a valid 2D coordinate"""
    assert isinstance(coord, np.ndarray), "bbx needs to be a numpy array"
    assert coord.shape == (2, ), "coord needs to 2D vector but got shape {}".format(coord.shape)

def assert_rotation(R):
    """Assert if coord is a valid 2D coordinate"""
    assert isinstance(R, np.ndarray), "R needs to be a numpy array"
    assert is_rotation_matrix(R)
