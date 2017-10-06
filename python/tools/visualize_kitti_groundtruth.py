#!/usr/bin/env python
"""visualizes kitti grundtruth annotations"""

import os.path as osp
import numpy as np
import cv2
import _init_paths
import RenderAndCompare as rac


def drawBbx3D(image, corners2d):
    """draw 3d bbx on image"""
    corners2d = np.floor(corners2d).astype(int)
    cv2.line(image, tuple(corners2d[:, 0]), tuple(corners2d[:, 1]), (0, 0, 255), 1)
    cv2.line(image, tuple(corners2d[:, 2]), tuple(corners2d[:, 3]), (0, 0, 255), 1)
    cv2.line(image, tuple(corners2d[:, 4]), tuple(corners2d[:, 5]), (0, 0, 255), 1)
    cv2.line(image, tuple(corners2d[:, 6]), tuple(corners2d[:, 7]), (0, 0, 255), 1)

    cv2.line(image, tuple(corners2d[:, 0]), tuple(corners2d[:, 2]), (0, 255, 0), 1)
    cv2.line(image, tuple(corners2d[:, 1]), tuple(corners2d[:, 3]), (0, 255, 0), 1)
    cv2.line(image, tuple(corners2d[:, 4]), tuple(corners2d[:, 6]), (0, 255, 0), 1)
    cv2.line(image, tuple(corners2d[:, 5]), tuple(corners2d[:, 7]), (0, 255, 0), 1)

    cv2.line(image, tuple(corners2d[:, 0]), tuple(corners2d[:, 4]), (255, 0, 0), 1)
    cv2.line(image, tuple(corners2d[:, 1]), tuple(corners2d[:, 5]), (255, 0, 0), 1)
    cv2.line(image, tuple(corners2d[:, 2]), tuple(corners2d[:, 6]), (255, 0, 0), 1)
    cv2.line(image, tuple(corners2d[:, 3]), tuple(corners2d[:, 7]), (255, 0, 0), 1)


def main():
    """main function"""
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

    paused = True
    fwd = True

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
        cam2_center = -np.linalg.inv(K).dot(P2[:, 3])
        velo_T_cam0 = rac.datasets.get_kitti_cam0_to_velo(calib_data)

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

            obj_pose = rac.datasets.get_kitti_object_pose(obj, velo_T_cam0, cam2_center)
            obj_center_proj = rac.geometry.project_point(K, obj_pose * (np.array([0., 0., 0.]))).astype(int)
            obj_x_proj = rac.geometry.project_point(K, obj_pose * np.array([1., 0., 0.])).astype(int)
            obj_y_proj = rac.geometry.project_point(K, obj_pose * np.array([0., 1., 0.])).astype(int)
            obj_z_proj = rac.geometry.project_point(K, obj_pose * np.array([0., 0., 1.])).astype(int)

            cv2.line(image, tuple(obj_center_proj), tuple(obj_x_proj), (0, 0, 255), 2)
            cv2.line(image, tuple(obj_center_proj), tuple(obj_y_proj), (0, 255, 0), 2)
            cv2.line(image, tuple(obj_center_proj), tuple(obj_z_proj), (255, 0, 0), 2)

            cv2.circle(image, tuple(obj_center_proj), 4, (0, 255, 255), -1)

            corners3D = rac.datasets.get_kitti_3D_bbox_corners(obj, obj_pose)

            corners2D = K.dot(corners3D)
            corners2D[0, :] = corners2D[0, :] / corners2D[2, :]
            corners2D[1, :] = corners2D[1, :] / corners2D[2, :]
            corners2D = corners2D[:2, :]

            drawBbx3D(image, corners2D)

            amodal_bbx = np.hstack((corners2D.min(axis=1), corners2D.max(axis=1)))
            amodal_bbx = np.floor(amodal_bbx).astype(int)

            cv2.rectangle(image,
                          (amodal_bbx[0], amodal_bbx[1]),
                          (amodal_bbx[2], amodal_bbx[3]),
                          (255, 0, 255), 1)

        cv2.imshow('image', image)

        key = cv2.waitKey(not paused)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key in [82, 83, 100, 119, 61, 43]:
            fwd = True
        elif key in [81, 84, 97, 115, 45]:
            fwd = False
        elif key == ord('p'):
            paused = not paused        
        i = i+1 if fwd else i-1


if __name__ == '__main__':
    main()
