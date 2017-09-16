"""
Standard Geometry routines
"""
from math import atan2, sqrt
import numpy as np


def wrap_to_pi(radians):
    """Wrap an angle (in radians) to [-pi, pi)"""
    # wrap to [0..2*pi]
    wrapped = radians % (2 * np.pi)
    # wrap to [-pi..pi]
    if wrapped > np.pi:
        wrapped -= 2 * np.pi

    return wrapped


def is_rotation_matrix(R, atol=1e-6):
    """Checks if a matrix is a valid rotation matrix"""
    assert R.shape == (3, 3), "R is not a 3x3 matrix. R.shape = {}".format(R.shape)
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < atol


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
    assert is_rotation_matrix(R)
    return R


def rotationX(angle):
    """Get the rotation matrix for rotation around X"""
    c = np.cos(angle)
    s = np.sin(angle)
    Rx = np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])
    return Rx


def rotationY(angle):
    """Get the rotation matrix for rotation around X"""
    c = np.cos(angle)
    s = np.sin(angle)
    Ry = np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])
    return Ry


def rotationZ(angle):
    """Get the rotation matrix for rotation around X"""
    c = np.cos(angle)
    s = np.sin(angle)
    Rz = np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])
    return Rz


def rotation_from_viewpoint(vp):
    """Get rotation matrix from viewpoint [azimuth, elevation, tilt]"""
    assert vp.shape == (3,)
    assert -np.pi <= vp[0] <= np.pi
    assert -np.pi / 2 <= vp[1] <= np.pi / 2
    assert -np.pi <= vp[2] <= np.pi

    R = rotationZ(-vp[2] - np.pi / 2).dot(rotationY(vp[1] + np.pi / 2)).dot(rotationZ(-vp[0]))
    assert is_rotation_matrix(R)
    return R


def viewpoint_from_rotation(R):
    """Returns viewpoint [azimuth, elevation, tilt] from rotation matrix"""
    assert is_rotation_matrix(R)

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
    tilt = wrap_to_pi(-res[0] - np.pi / 2)

    return np.array([azimuth, elevation, tilt])


def eulerZYX_from_rotation(R):
    """Calculates rotation matrix to euler angles (ZYX)"""

    assert is_rotation_matrix(R)

    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class Pose(object):
    """Pose class [R | t]"""

    def __init__(self, R=np.eye(3), t=np.zeros(3)):
        """Initilize from 3x3 rotation matrix and vector3"""
        assert R.shape == (3, 3)
        assert t.shape == (3, )
        self.R = R
        self.t = t

    def __repr__(self):
        """returns an unique string repr of self"""
        repr_str = str(self.matrix())
        return 'Pose [R | t] = ' + '               '.join(repr_str.splitlines(True))

    def matrix(self):
        """returns as 3x4 [R | t] matrix"""
        return np.hstack((self.R, self.t.reshape((3, 1))))

    def inverse(self):
        """returns inverse of self"""
        return Pose(self.R.T, - self.R.T.dot(self.t))

    def __mul__(self, other):
        """
        Return the product of self with other i.e out = self * other
        """
        if isinstance(other, np.ndarray):
            assert other.shape[0] == 3, "expects leading dimension of other to be 3. other.shape = {}".format(other.shape)
            assert other.ndim == 1 or other.ndim == 2, "other.shape = {}".format(other.shape)
            projected = self.R.dot(other.reshape(3, other.shape[1] if other.ndim == 2 else 1)) + self.t.reshape((3, 1))
            return np.squeeze(projected)
        elif isinstance(other, Pose):
            return Pose(self.R.dot(other.R), self.R.dot(other.t) + self.t)
        raise TypeError('This operation is only supported with numpy array or Pose objects')
