"""
KITTI Helper functions
"""

from math import atan

import numpy as np

from ..geometry import (Pose, eulerZYX_from_rotation, is_rotation_matrix,
                        rotationZ, wrap_to_pi)


def write_kitti_object_labels(objects, filepath):
    """
    Given a list of object and  a filename, it writes the list of objects in
    kitti format to the  filename
    Each object is ecpected to be a dict with atleast three fields:
    'type':  like 'Car', 'Pedestrian', or 'Cyclist'
    'bbox':  a list of 4 numbers specifying [left, top, right, bottom] for bbox pixel coordinates (0 based index)
    'score' float indicating confidence in detection

    See the read_kitti_object_labels() for the full list of keys. If the other fields other not present,
    some default magic number are written so that kitti evaluation server understands it.
    """
    with open(filepath, "w") as f:
        for obj in objects:
            f.write('{} '.format(obj['type']))
            if 'truncation' in obj:
                f.write('{} '.format(obj['truncation']))
            else:
                f.write('-1 ')
            if 'occlusion' in obj:
                f.write('{} '.format(obj['occlusion']))
            else:
                f.write('-1 ')
            if 'alpha' in obj:
                assert -np.pi <= obj['alpha'] <= np.pi
                f.write('{} '.format(obj['alpha']))
            else:
                f.write('-10 ')
            bbx = obj['bbox']
            assert len(bbx) == 4
            f.write('{} {} {} {} '.format(bbx[0], bbx[1], bbx[2], bbx[3]))
            if 'dimension' in obj:
                dimension = obj['dimension']
                assert len(dimension) == 3
                f.write('{} {} {} '.format(dimension[0], dimension[1], dimension[2]))
            else:
                f.write('-1 -1 -1 ')
            if 'location' in obj:
                location = obj['location']
                assert len(location) == 3
                f.write('{} {} {} '.format(location[0], location[1], location[2]))
            else:
                f.write('-1000 -1000 -1000 ')
            if 'rotation_y' in obj:
                assert -np.pi <= obj['rotation_y'] <= np.pi
                f.write('{} '.format(obj['rotation_y']))
            else:
                f.write('-10 ')
            f.write('{} \n'.format(obj['score']))


def read_kitti_object_labels(filepath):
    """
    Reads a KITTI style label file to list of objects
    Each label file is for all the objects in one singele image
    Each object in the returned list of objects is a dict with the
    keys: type, truncation, occlusion, alpha, bbox, dimension, location, rotation_y, and, score
    """
    with open(filepath, "r") as f:
        non_empty_lines = [line.strip() for line in f if line.strip()]

    objinfos = [line.strip().split(' ') for line in non_empty_lines]

    objects = []

    for objinfo in objinfos:
        assert len(objinfo) == 15 or len(objinfo) == 16, 'Found {} tokens in Line: "{}". Expects only 15/16 token'.format(len(objinfo), objinfo)
        obj = {}
        obj['type'] = objinfo[0]
        if objinfo[1] != '-1':
            obj['truncation'] = float(objinfo[1])
        if objinfo[2] != '-1':
            obj['occlusion'] = int(objinfo[2])
        if objinfo[3] != '-10':
            obj['alpha'] = float(objinfo[3])
        obj['bbox'] = [float(x) for x in objinfo[4:8]]
        if objinfo[8:11] != ['-1', '-1', '-1']:
            obj['dimension'] = [float(x) for x in objinfo[8:11]]
        if objinfo[11:14] != ['-1000', '-1000', '-1000']:
            obj['location'] = [float(x) for x in objinfo[11:14]]
        if objinfo[14] != '-10':
            obj['rotation_y'] = float(objinfo[14])
        if len(objinfo) == 16:
            obj['score'] = float(objinfo[15])
        else:
            obj['score'] = 1.0
        objects.append(obj)
    return objects


def read_kitti_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        # Get non whitespace-only lines
        lines = filter(None, (line.rstrip() for line in f))
        for line in lines:
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def get_kitti_cam0_to_velo(calib_data):
    """Get R, t for cam0 to velo s.t x_velo = [R | t] x_cam0"""
    assert calib_data['R0_rect'].shape == (9,)
    assert calib_data['Tr_velo_to_cam'].shape == (12,)
    R0_rect = calib_data['R0_rect'].reshape((3, 3))
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape((3, 4))
    cam_R_velo = R0_rect.dot(Tr_velo_to_cam[:, :3])
    cam_t_velo = R0_rect.dot(Tr_velo_to_cam[:, 3])
    R = cam_R_velo.T
    t = - cam_R_velo.T.dot(cam_t_velo)
    assert is_rotation_matrix(R)
    return Pose(R, t)


def get_kitti_velo_to_cam(calib_data, cam_center=np.zeros(3)):
    """Get R, t for velo to cam s.t x_cam = [R | t] x_velo"""
    assert calib_data['R0_rect'].shape == (9,)
    assert calib_data['Tr_velo_to_cam'].shape == (12,)
    R0_rect = calib_data['R0_rect'].reshape((3, 3))
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape((3, 4))
    cam_R_velo = R0_rect.dot(Tr_velo_to_cam[:, :3])
    cam_t_velo = R0_rect.dot(Tr_velo_to_cam[:, 3]) - cam_center
    assert is_rotation_matrix(cam_R_velo)
    return Pose(cam_R_velo, cam_t_velo)


def get_kitti_object_pose(obj, velo_T_cam0, cam_center=np.zeros(3)):
    """Get object pose (rotation and translation)"""
    phi_cam = obj['rotation_y']
    phi_velo = wrap_to_pi(-phi_cam - np.pi / 2)
    velo_R_obj = rotationZ(phi_velo)
    velo_t_obj = velo_T_cam0 * np.array(obj['location']) + np.array([0, 0, obj['dimension'][0] / 2.0])
    cam_T_obj = Pose(t=-cam_center) * (velo_T_cam0.inverse() * Pose(velo_R_obj, velo_t_obj))
    return cam_T_obj


def get_kitti_3D_bbox_corners(obj, pose):
    """Get the 3D corners of the bounding box for a kitti object"""
    assert is_rotation_matrix(pose.R)

    H = obj['dimension'][0]
    W = obj['dimension'][1]
    L = obj['dimension'][2]

    x_corners = np.array([-L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2])
    y_corners = np.array([-W / 2, -W / 2, W / 2, W / 2, -W / 2, -W / 2, W / 2, W / 2])
    z_corners = np.array([-H / 2, -H / 2, -H / 2, -H / 2, H / 2, H / 2, H / 2, H / 2])

    corners3D = pose * np.vstack((x_corners, y_corners, z_corners))
    return corners3D


def get_kitti_amodal_bbx(obj, K, obj_pose):
    """Get amodal box by back projecting 3D bounding box of the object"""
    corners3D = get_kitti_3D_bbox_corners(obj, obj_pose)
    corners2D = K.dot(corners3D)
    corners2D[0, :] = corners2D[0, :] / corners2D[2, :]
    corners2D[1, :] = corners2D[1, :] / corners2D[2, :]
    corners2D = corners2D[:2, :]
    amodal_bbx = np.hstack((corners2D.min(axis=1), corners2D.max(axis=1)))
    return amodal_bbx


def get_kitti_alpha_from_object_pose(cam_T_obj, velo_T_cam):
    """Get alpha from object_pose"""
    assert is_rotation_matrix(cam_T_obj.R)
    assert is_rotation_matrix(velo_T_cam.R)

    velo_T_obj = velo_T_cam * cam_T_obj
    beta = atan(velo_T_obj.t[1] / velo_T_obj.t[0])
    phi_velo = eulerZYX_from_rotation(velo_T_obj.R)[2]
    phi_cam = wrap_to_pi(-phi_velo - np.pi / 2)
    alpha = wrap_to_pi(phi_cam + beta)
    return alpha
