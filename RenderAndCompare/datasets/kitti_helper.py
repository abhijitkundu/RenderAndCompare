import numpy as np


def read_kitti_object_label_file(filepath):
    objinfos = [line.strip().split(' ') for line in open(filepath).readlines()]

    objects = []

    for objinfo in objinfos:
        assert len(objinfo) == 15
        obj = {}
        obj['type'] = objinfo[0]
        obj['truncated'] = float(objinfo[1])
        obj['occluded'] = int(objinfo[2])
        obj['alpha'] = float(objinfo[3])
        obj['bbox'] = [float(x) for x in objinfo[4:8]]
        obj['dimension'] = [float(x) for x in objinfo[8:11]]
        obj['location'] = [float(x) for x in objinfo[11:14]]
        obj['rotation_y'] = float(objinfo[14])
        objects.append(obj)
    return objects

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


def get_kitti_3D_bbox_corners(object):
    c = np.cos(object['rotation_y'])
    s = np.sin(object['rotation_y'])
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])

    W = object['dimension'][1]
    H = object['dimension'][0]
    L = object['dimension'][2]

    x_corners = np.array([L / 2, L / 2, -L / 2, -L / 2, L / 2, L / 2, -L / 2, -L / 2])
    y_corners = np.array([0, 0, 0, 0, -H, -H, -H, -H])
    z_corners = np.array([W / 2, -W / 2, -W / 2, W / 2, W / 2, -W / 2, -W / 2, W / 2])

    corners3D = R.dot(np.vstack((x_corners, y_corners, z_corners)))
    corners3D = corners3D + np.array(object['location']).reshape((3, 1))
    return corners3D
