#!/usr/bin/env python

import _init_paths
import cv2
import numpy as np
import os.path as osp
import RenderAndCompare as rac


kitti_object_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
assert osp.exists(kitti_object_dir), 'KITTI-Object dir "{}" dsoes not exist'.format(kitti_object_dir)

label_dir = osp.join(kitti_object_dir, 'training', 'label_2_updated')
image_dir = osp.join(kitti_object_dir, 'training', 'image_2')
calib_dir = osp.join(kitti_object_dir, 'training', 'calib')

# categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']
categories = ['Car', 'Van']


assert osp.exists(label_dir)
assert osp.exists(image_dir)
assert osp.exists(calib_dir)

# num_of_images = 7481
num_of_images = 7481

min_height = 20  # minimum height for evaluated groundtruth/detections
max_occlusion = 2  # maximum occlusion level of the groundtruth used for evaluation
max_truncation = 0.7  # maximum truncation level of the groundtruth used for evaluation


cv2.namedWindow('image', cv2.WINDOW_NORMAL)

i = 0
while True:
    i = max(0, min(i, num_of_images - 1))
    cv2.displayOverlay('image', 'Image: {} / {}'.format(i, num_of_images - 1))

    base_name = '%06d' % (i)
    image_file_path = osp.join(image_dir, base_name + '.png')
    label_file_path = osp.join(label_dir, base_name + '.txt')
    calib_file_path = osp.join(calib_dir, base_name + '.txt')

    assert osp.exists(image_file_path)
    assert osp.exists(label_file_path)
    assert osp.exists(calib_file_path)

    image = cv2.imread(image_file_path)
    objects = rac.datasets.read_kitti_object_labels(label_file_path)

    calib_data = rac.datasets.read_kitti_calib_file(calib_file_path)
    P1 = calib_data['P1'].reshape((3, 4))
    P2 = calib_data['P2'].reshape((3, 4))

    K = P1[:3, :3]
    assert np.all(P2[:3, :3] == K)
    cam_center = -np.linalg.inv(K).dot(P2[:, 3])

    filtered_objects = []
    for obj in objects:
        if obj['type'] not in categories:
            continue

        bbx = np.asarray(obj['bbox'])

        too_hard = False
        if (bbx[3] - bbx[1]) < min_height:
            too_hard = True
        if obj['occlusion'] > max_occlusion:
            too_hard = True
        if obj['truncation'] > max_truncation:
            too_hard = True

        if not too_hard:
            filtered_objects.append(obj)

    for obj in filtered_objects:
        bbx = np.floor(np.asarray(obj['bbox'])).astype(int)

        cv2.rectangle(image,
                      (bbx[0], bbx[1]),
                      (bbx[2], bbx[3]),
                      (0, 255, 0), 1)
        # bbx_str = '{} Occ:{} Trunc{:0.2f}'.format(obj['type'], obj['occlusion'], obj['truncation'])
        # cv2.putText(image, bbx_str, (bbx[0] + 5, bbx[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        R, t = rac.datasets.get_kitti_object_pose(obj, cam_center)
        obj_center_proj = rac.geometry.project_point(K, R.dot(np.array([0., 0., 0.])) + t).astype(int)
        obj_x_proj = rac.geometry.project_point(K, R.dot(np.array([1., 0., 0.])) + t).astype(int)
        obj_y_proj = rac.geometry.project_point(K, R.dot(np.array([0., 1., 0.])) + t).astype(int)
        obj_z_proj = rac.geometry.project_point(K, R.dot(np.array([0., 0., 1.])) + t).astype(int)

        cv2.line(image, tuple(obj_center_proj), tuple(obj_x_proj), (0, 0, 255), 2)
        cv2.line(image, tuple(obj_center_proj), tuple(obj_y_proj), (0, 255, 0), 2)
        cv2.line(image, tuple(obj_center_proj), tuple(obj_z_proj), (255, 0, 0), 2)

        cv2.circle(image, tuple(obj_center_proj), 4, (0, 255, 255), -1)

        amodal_bbx = rac.datasets.get_kitti_amodal_bbx(obj, K, cam_center)
        amodal_bbx = np.floor(amodal_bbx).astype(int)

        cv2.rectangle(image,
                      (amodal_bbx[0], amodal_bbx[1]),
                      (amodal_bbx[2], amodal_bbx[3]),
                      (255, 0, 255), 1)

    cv2.imshow('image', image)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        break
    elif key in [82, 83, 100, 119, 61, 43]:
        i += 1
    elif key in [81, 84, 97, 115, 45]:
        i -= 1
