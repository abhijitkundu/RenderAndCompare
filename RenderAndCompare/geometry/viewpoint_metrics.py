import numpy as np
from scipy.linalg import logm, norm
import math


# Calculates Rotation Matrix given euler angles (Rx Ry Rz)
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_x, np.dot(R_y, R_z))

    return R


def anglesTodcm(theta):
    dcm = np.zeros((3, 3))

    cang = [math.cos(x) for x in theta]
    sang = [math.sin(x) for x in theta]

    dcm[0, 0] = cang[1] * cang[0]
    dcm[0, 1] = cang[1] * sang[0]
    dcm[0, 2] = -sang[1]
    dcm[1, 0] = sang[2] * sang[1] * cang[0] - cang[2] * sang[0]
    dcm[1, 1] = sang[2] * sang[1] * sang[0] + cang[2] * cang[0]
    dcm[1, 2] = sang[2] * cang[1]
    dcm[2, 0] = cang[2] * sang[1] * cang[0] + sang[2] * sang[0]
    dcm[2, 1] = cang[2] * sang[1] * sang[0] - sang[2] * cang[0]
    dcm[2, 2] = cang[2] * cang[1]

    return dcm


def viewpoint2rotation(viewpoint):
    assert len(viewpoint) == 2, 'Expects viewpoint to be 3 dimensional tuple/list'
    theta = [math.radians(x) for x in viewpoint]
    return eulerAnglesToRotationMatrix(theta)


def viewpoint2dcm(viewpoint):
    assert len(viewpoint) == 3, 'Expects viewpoint to be 3 dimensional tuple/list'
    theta = [math.radians(x) for x in viewpoint]
    return anglesTodcm(theta)


def compute_geodesic_errors(gt_vps, pred_vps):
    assert len(gt_vps) == len(pred_vps)
    num_of_data_points = len(gt_vps)
    geodesic_errors = np.zeros(num_of_data_points)
    for i in xrange(num_of_data_points):
        R_gt = viewpoint2dcm(gt_vps[i])
        R_pred = viewpoint2dcm(pred_vps[i])
        geodesic_errors[i] = norm(logm(np.transpose(R_pred).dot(R_gt)), 2) / math.sqrt(2)
    return geodesic_errors
