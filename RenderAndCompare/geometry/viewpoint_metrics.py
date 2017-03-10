import numpy as np
from scipy.linalg import logm, norm
import math


def azimuth_to_alpha(azimuth):
    # Add offset of 90  and then reduce the angle
    angle = (azimuth + 90) % 360.0
    # force it to be the positive remainder, so that 0 <= angle < 360
    angle = (angle + 360) % 360.0

    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    if angle > 180:
        angle -= 360
    return math.radians(angle)


def alpha_to_azimuth(alpha):
    assert -np.pi <= alpha <= np.pi
    alpha_deg = math.degrees(alpha)
    azimuth = (alpha_deg - 90) % 360.0
    return azimuth


def anglesTodcmZYX(theta):
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


def anglesTodcmZXZ(theta):
    dcm = np.zeros((3, 3))

    cang = [math.cos(x) for x in theta]
    sang = [math.sin(x) for x in theta]

    dcm[0, 0] = -sang[0] * cang[1] * sang[2] + cang[0] * cang[2]
    dcm[0, 1] = cang[0] * cang[1] * sang[2] + sang[0] * cang[2]
    dcm[0, 2] = sang[1] * sang[2]
    dcm[1, 0] = -sang[0] * cang[2] * cang[1] - cang[0] * sang[2]
    dcm[1, 1] = cang[0] * cang[2] * cang[1] - sang[0] * sang[2]
    dcm[1, 2] = sang[1] * cang[2]
    dcm[2, 0] = sang[0] * sang[1]
    dcm[2, 1] = -cang[0] * sang[1]
    dcm[2, 2] = cang[1]

    return dcm


def viewpoint2dcm(viewpoint):
    assert len(viewpoint) == 3, 'Expects viewpoint to be 3 dimensional tuple/list'
    theta = [math.radians(x) for x in viewpoint]
    return anglesTodcmZYX(theta)


def angle_error_render4cnn(gt_vp, pred_vp):
    R_gt = viewpoint2dcm(gt_vp)
    R_pred = viewpoint2dcm(pred_vp)
    return norm(logm(np.transpose(R_pred).dot(R_gt)), 2) / math.sqrt(2)


def angle_error_vpkp(gt_vp, pred_vp):
    theta_gt = [math.radians(x) for x in gt_vp]
    theta_pred = [math.radians(x) for x in pred_vp]
    R_gt = anglesTodcmZXZ([theta_gt[0], theta_gt[1] - np.pi / 2.0, -theta_gt[2]])
    R_pred = anglesTodcmZXZ([theta_pred[0], theta_pred[1] - np.pi / 2.0, -theta_pred[2]])
    return norm(logm(R_gt.dot(np.transpose(R_pred))), ord='fro') / math.sqrt(2)


def compute_geodesic_errors(gt_vps, pred_vps):
    assert len(gt_vps) == len(pred_vps)
    num_of_data_points = len(gt_vps)
    geodesic_errors = np.zeros(num_of_data_points)
    for i in xrange(num_of_data_points):
        geodesic_errors[i] = angle_error_render4cnn(gt_vps[i], pred_vps[i])
    return geodesic_errors
