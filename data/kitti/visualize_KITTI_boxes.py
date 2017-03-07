import cv2
import numpy as np
import os.path as osp


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


def compute3Dbbox(object):
    c = np.cos(object['rotation_y'])
    s = np.sin(object['rotation_y'])
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])

    l = object['dimension'][2]
    w = object['dimension'][1]
    h = object['dimension'][0]

    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners3D = R.dot(np.vstack((x_corners, y_corners, z_corners)))
    corners3D = corners3D + np.array(object['location']).reshape((3, 1))
    return corners3D


KITTI_OBJECT_DIR = '/media/Scratchspace/KITTI-Object'

label_dir = osp.join(KITTI_OBJECT_DIR, 'training', 'label_2')
image_dir = osp.join(KITTI_OBJECT_DIR, 'training', 'image_2')
calib_dir = osp.join(KITTI_OBJECT_DIR, 'training', 'calib')
# num_of_images = 7481
num_of_images = 7481

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

for i in xrange(num_of_images):
    print 'Working on image: {} / {}'.format(i, num_of_images)

    base_name = '%06d' % (i)
    image_file_path = osp.join(image_dir, base_name + '.png')
    label_file_path = osp.join(label_dir, base_name + '.txt')
    calib_file_path = osp.join(calib_dir, base_name + '.txt')

    assert osp.exists(image_file_path)
    assert osp.exists(label_file_path)
    assert osp.exists(calib_file_path)

    image = cv2.imread(image_file_path)
    objects = read_kitti_object_label_file(label_file_path)

    P = read_kitti_calib_file(calib_file_path)['P2'].reshape((3, 4))

    for obj in objects:
        if obj['type'] != 'Car':
            continue

        bbx = np.floor(np.asarray(obj['bbox'])).astype(int)
        cv2.rectangle(image,
                      (bbx[0], bbx[1]),
                      (bbx[2], bbx[3]),
                      (0, 255, 0), 1)

        corners3D = compute3Dbbox(obj)

        corners2D = P.dot(np.vstack((corners3D, np.ones(8))))
        corners2D[0, :] = corners2D[0, :] / corners2D[2, :]
        corners2D[1, :] = corners2D[1, :] / corners2D[2, :]
        corners2D = corners2D[:2, :]

        min_bbx = np.floor(corners2D.min(axis=1)).astype(int)
        max_bbx = np.floor(corners2D.max(axis=1)).astype(int)

        cv2.rectangle(image,
                      (min_bbx[0], min_bbx[1]),
                      (max_bbx[0], max_bbx[1]),
                      (255, 0, 255), 1)

    cv2.imshow('image', image)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        break
