"""
The script contains functions to compute geodesic error in viewpoint.abs
The implementation exactly follows the Render4CNN paper
"""
import math

import numpy as np
from scipy.linalg import logm, norm

from RenderAndCompare.geometry import assert_viewpoint

def compute_viewpoint_error(vpA, vpB, use_degrees=True):
    """compute viewpoint error (absolute) in degrees (default) or radians (if use_degrees=False)"""
    angle_error = (np.array(vpA) - np.array(vpB) + np.pi) % (2 * np.pi) - np.pi
    assert_viewpoint(angle_error)
    if use_degrees:
        angle_error = np.degrees(angle_error)
    return np.fabs(angle_error)

def compute_geodesic_error(gt_vp, pred_vp, use_degrees=True):
    """compute geodesic viewpoint error in degrees (default) or radians (if use_degrees=False)"""
    geodesic_error = geodesic_error_render4cnn(gt_vp, pred_vp)
    if use_degrees:
        geodesic_error = np.degrees(geodesic_error)
    return geodesic_error


def viewpoint_to_dcmZYX(vp):
    """convert viewpoint to matlab style rot matrix with 'ZYX' order"""
    dcm = np.zeros((3, 3))

    cang = np.cos(vp)
    sang = np.sin(vp)

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


def viewpoint_to_dcmZXZ(vp):
    """convert viewpoint to matlab style rot matrix with 'ZXZ' order"""
    dcm = np.zeros((3, 3))

    cang = np.cos(vp)
    sang = np.sin(vp)

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

def geodesic_error_render4cnn(gt_vp, pred_vp):
    """geodesic viewpoint error Render4CNN paper style"""
    R_gt = viewpoint_to_dcmZYX(gt_vp)
    R_pred = viewpoint_to_dcmZYX(pred_vp)
    return norm(logm(np.transpose(R_pred).dot(R_gt)), 2) / math.sqrt(2)


def geodesic_error_vpkp(gt_vp, pred_vp):
    """geodesic viewpoint error VpKp paper style"""
    R_gt = viewpoint_to_dcmZXZ([gt_vp[0], gt_vp[1] - np.pi / 2.0, -gt_vp[2]])
    R_pred = viewpoint_to_dcmZXZ([pred_vp[0], pred_vp[1] - np.pi / 2.0, -pred_vp[2]])
    return norm(logm(R_gt.dot(np.transpose(R_pred))), ord='fro') / math.sqrt(2)
