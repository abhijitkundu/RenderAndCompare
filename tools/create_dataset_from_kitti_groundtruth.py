#!/usr/bin/env python

import _init_paths
import os.path as osp
from os import makedirs
import RenderAndCompare as rac
from collections import OrderedDict
import numpy as np
import math
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    kitti_object_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    assert osp.exists(kitti_object_dir), 'KITTI Object dir "{}" does not exist'.format(kitti_object_dir)

    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', 'trainval.txt')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split_file", default=splits_file_default, help="Path to split file")
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder to save the cropped images")
    parser.add_argument("-a", "--augment", type=int, default=0, help="Number of augmentations per gt box")
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.add_argument('--no_flip', dest='flip', action='store_false')
    parser.set_defaults(flip=False)
    args = parser.parse_args()

    assert osp.exists(args.split_file), 'Path to split file does not exist: {}'.format(args.split_file)
    image_names = [x.rstrip() for x in open(args.split_file)]
    num_of_images = len(image_names)
    print 'Using Split {} with {} images'.format(osp.basename(args.split_file), num_of_images)

    label_dir = osp.join(kitti_object_dir, 'training', 'label_2')
    image_dir = osp.join(kitti_object_dir, 'training', 'image_2')
    calib_dir = osp.join(kitti_object_dir, 'training', 'calib')

    assert osp.exists(label_dir)
    assert osp.exists(image_dir)
    assert osp.exists(calib_dir)

    if not osp.exists(args.output_folder):
        print 'Created new output directory at {}'.format(args.output_folder)
        makedirs(args.output_folder)
    else:
        print 'Will overwrite contents in {}'.format(args.output_folder)

    dataset_name = 'kitti_' + osp.splitext(osp.basename(args.split_file))[0]
    dataset = rac.datasets.Dataset(dataset_name)
    dataset.set_rootdir(args.output_folder)

    min_height = 25  # minimum height for evaluated groundtruth/detections
    max_occlusion = 2  # maximum occlusion level of the groundtruth used for evaluation
    max_truncation = 0.5  # maximum truncation level of the groundtruth used for evaluation

    for image_name in tqdm(image_names):
        image_file_path = osp.join(image_dir, image_name + '.png')
        label_file_path = osp.join(label_dir, image_name + '.txt')
        calib_file_path = osp.join(calib_dir, image_name + '.txt')

        assert osp.exists(image_file_path)
        assert osp.exists(label_file_path)
        assert osp.exists(calib_file_path)

        objects = rac.datasets.read_kitti_object_labels(label_file_path)

        filtered_objects = []
        for obj in objects:
            if obj['type'] not in ['Car']:
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

        if len(filtered_objects) == 0:
            continue

        image = cv2.imread(image_file_path)
        W = image.shape[1]
        H = image.shape[0]
        calib_data = rac.datasets.read_kitti_calib_file(calib_file_path)
        P1 = calib_data['P1'].reshape((3, 4))
        P2 = calib_data['P2'].reshape((3, 4))

        K = P1[:3, :3]
        assert np.all(P2[:3, :3] == K)
        cam2_center = -np.linalg.inv(K).dot(P2[:, 3])
        principal_point = K[:2, 2]

        for i, obj in enumerate(filtered_objects):
            crop_bbx = np.array(obj['bbox'])
            amodal_bbx = rac.datasets.get_kitti_amodal_bbx(obj, P2)

            bbx_min = np.maximum.reduce([np.array([0, 0]), np.floor(crop_bbx[:2]).astype(int)])
            bbx_max = np.minimum.reduce([np.array([W - 1, H - 1]), np.floor(crop_bbx[2:]).astype(int)])

            # Bad BBX
            if np.any((bbx_max - bbx_min) == 0):
                continue

            cropped_image = image[bbx_min[1]:bbx_max[1] + 1, bbx_min[0]:bbx_max[0] + 1, :]
            assert cropped_image.ndim == 3

            # make sure my crop_bbx is updated with the final cropped image size
            crop_bbx[:2] = bbx_min
            crop_bbx[2:] = bbx_max + 1.0

            # Save cropped image with some filename
            cropped_image_name = '{}_{:04d}_aug{:02d}.png'.format(image_name, i, 0, )
            cv2.imwrite(osp.join(args.output_folder, cropped_image_name), cropped_image)

            azimuth = rac.geometry.alpha_to_azimuth(obj['alpha'])
            obj_center = np.array(obj['location']) - np.array([0, obj['dimension'][0] / 2.0, 0])
            obj_center_cam2 = obj_center - cam2_center
            obj_center_cam2_xz_proj = np.array([obj_center_cam2[0], 0, obj_center_cam2[2]])
            elevation_angle = np.arctan2(np.linalg.norm(np.cross(obj_center_cam2, obj_center_cam2_xz_proj)), np.dot(obj_center_cam2, obj_center_cam2_xz_proj))
            if obj_center_cam2[1] < 0:
                elevation_angle = -elevation_angle
            elevation = math.degrees(elevation_angle) % 360
            distance = np.linalg.norm(obj_center_cam2)

            # subtract principal point
            crop_bbx[:2] -= principal_point
            crop_bbx[2:] -= principal_point
            amodal_bbx[:2] -= principal_point
            amodal_bbx[2:] -= principal_point

            annotation = OrderedDict()
            annotation['image_file'] = cropped_image_name
            annotation['viewpoint'] = [azimuth, elevation, 0, distance]
            annotation['bbx_amodal'] = [amodal_bbx[0], amodal_bbx[1], amodal_bbx[2] - amodal_bbx[0], amodal_bbx[3] - amodal_bbx[1]]
            annotation['bbx_crop'] = [crop_bbx[0], crop_bbx[1], crop_bbx[2] - crop_bbx[0], crop_bbx[3] - crop_bbx[1]]
            dataset.add_annotation(annotation)

    print 'Finished creating dataset with {} annotations'.format(dataset.num_of_annotations())
    dataset.write_data_to_json(osp.join(osp.dirname(args.output_folder), dataset_name + '.json'))
