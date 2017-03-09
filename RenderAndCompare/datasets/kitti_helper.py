import numpy as np


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
            f.write('%s '.format(obj['type']))
            if 'truncation' in obj:
                f.write('%.2f '.format(obj['truncation']))
            else:
                f.write('-1 ')
            if 'occlusion' in obj:
                f.write('%d '.format(obj['occlusion']))
            else:
                f.write('-1 ')
            if 'alpha' in obj:
                assert -np.pi <= obj['alpha'] <= np.pi
                f.write('%.2f '.format(obj['alpha']))
            else:
                f.write('-10 ')
            bbx = obj['bbox']
            assert len(bbx) == 4
            f.write('%.2f %.2f %.2f %.2f '.format(bbx[0], bbx[1], bbx[2], bbx[3]))
            if 'dimension' in obj:
                dimension = obj['dimension']
                assert len(dimension) == 3
                f.write('%.2f %.2f %.2f '.format(dimension[0], dimension[1], dimension[2]))
            else:
                f.write('-1 -1 -1 ')
            if 'location' in obj:
                location = obj['location']
                assert len(location) == 3
                f.write('%.2f %.2f %.2f '.format(location[0], location[1], location[2]))
            else:
                f.write('-1000 -1000 -1000 ')
            if 'rotation_y' in obj:
                assert -np.pi <= obj['rotation_y'] <= np.pi
                f.write('%.2f '.format(obj['rotation_y']))
            else:
                f.write('-10 ')
            f.write('%.2f \n'.format(obj['score']))


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
        obj['truncation'] = float(objinfo[1])
        obj['occlusion'] = int(objinfo[2])
        obj['alpha'] = float(objinfo[3])
        obj['bbox'] = [float(x) for x in objinfo[4:8]]
        obj['dimension'] = [float(x) for x in objinfo[8:11]]
        obj['location'] = [float(x) for x in objinfo[11:14]]
        obj['rotation_y'] = float(objinfo[14])
        if len(objinfo) == 16:
            obj['score'] = float(objinfo[15])
        else:
            obj['score'] = 1.0
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
    """Get the 3D corners of the oriented bounding box for a kitti object"""
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
