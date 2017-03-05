import numpy as np
from scipy.linalg import logm, norm
import math


# Calculates Rotation Matrix given euler angles (Rz Ry Rx)
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

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def viewpoint2rotation(viewpoint):
    assert len(viewpoint) == 3, 'Expects viewpoint to be 3 dimensional tuple/list'
    theta = [math.radians(x) for x in viewpoint]
    return eulerAnglesToRotationMatrix(theta)


def compute_vp_acc(gt_vps, pred_vps):
    assert len(gt_vps) == len(pred_vps)
    num_of_data_points = len(gt_vps)
    geodesic_errors = np.zeros(num_of_data_points)
    for i in xrange(num_of_data_points):
        R_gt = viewpoint2rotation(gt_vps[i])
        R_pred = viewpoint2rotation(pred_vps[i])
        geodesic_errors[i] = norm(logm(np.transpose(R_gt).dot(R_pred))) / math.sqrt(2)

    thresh_angle = np.pi / 6.0
    acc_pi_by_6 = float((geodesic_errors < thresh_angle).sum()) / num_of_data_points
    med_err_deg = math.degrees(np.median(geodesic_errors))
    return acc_pi_by_6, med_err_deg
