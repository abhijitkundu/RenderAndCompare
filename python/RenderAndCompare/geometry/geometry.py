"""
Standard Geometry routines
"""
import numpy as np
from math import sqrt, atan2


def wrapToPi(radians):
    """Wrap an angle (in radians) to [-pi, pi)"""
    # wrap to [0..2*pi]
    wrapped = radians % np.pi
    # wrap to [-pi..pi]
    if wrapped > np.pi:
        wrapped -= 2 * np.pi

    return wrapped


def project_point(P, point_3D):
    """Projects a single 3D point to a 2D point"""
    if P.shape == (3, 4):
        point_2D = P.dot(np.append(point_3D, 1))
    elif P.shape == (3, 3):
        point_2D = P.dot(point_3D)
    else:
        raise ValueError('Expects 3x3 or 3x4 matrix')
    point_2D = point_2D[:-1] / point_2D[-1]
    assert point_2D.shape == (2,)
    return point_2D


def skew_symm(x):
    """Returns skew-symmetric matrix from vector of length 3"""
    assert x.shape == (3,)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rotation_from_two_vectors(a, b):
    """
    Returns rotation matrix that rotates a to be same as b
    See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    """
    assert a.shape == (3,)
    assert b.shape == (3,)
    au = a / np.linalg.norm(a)
    bu = b / np.linalg.norm(b)
    ssv = skew_symm(np.cross(au, bu))
    c = au.dot(bu)
    assert c != -1.0, 'Cannot handle case when a and b are exactly opposite'
    R = np.eye(3) + ssv + np.matmul(ssv, ssv) / (1 + c)
    return R


def get_viewpoint_from_rotation(R):
    """Returns viewpoint [azimuth, elevation, tilt] from rotation matrix"""
    assert R.shape == (3, 3)

    plusMinus = -1
    minusPlus = 1

    I = 2
    J = 1
    K = 0

    res = np.empty(3)

    Rsum = sqrt((R[I, J] * R[I, J] + R[I, K] * R[I, K] + R[J, I] * R[J, I] + R[K, I] * R[K, I]) / 2)
    res[1] = atan2(Rsum, R[I, I])

    # There is a singularity when sin(beta) == 0
    if Rsum > 4 * np.finfo(float).eps:
        # sin(beta) != 0
        res[0] = atan2(R[J, I], minusPlus * R[K, I])
        res[2] = atan2(R[I, J], plusMinus * R[I, K])
    elif R[I, I] > 0:
        # sin(beta) == 0 and cos(beta) == 1
        spos = plusMinus * R[K, J] + minusPlus * R[J, K]  # 2*sin(alpha + gamma)
        cpos = R[J, J] + R[K, K]                         # 2*cos(alpha + gamma)
        res[0] = atan2(spos, cpos)
        res[2] = 0
    else:
        # sin(beta) == 0 and cos(beta) == -1
        sneg = plusMinus * R[K, J] + plusMinus * R[J, K]  # 2*sin(alpha - gamma)
        cneg = R[J, J] - R[K, K]                         # 2*cos(alpha - gamma)
        res[0] = atan2(sneg, cneg)
        res[2] = 0

    azimuth = -res[2]
    elevation = res[1] - np.pi / 2
    tilt = wrapToPi(-res[0] - np.pi / 2)

    return np.array([azimuth, elevation, tilt])
