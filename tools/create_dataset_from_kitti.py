#!/usr/bin/env python

import sys
import os.path as osp
from os import makedirs
from collections import OrderedDict
import math
import numpy as np
import cv2
from tqdm import tqdm
import _init_paths
from RenderAndCompare.datasets import Dataset
from RenderAndCompare.datasets import read_kitti_calib_file
from RenderAndCompare.datasets import read_kitti_object_labels
from RenderAndCompare.datasets import get_kitti_amodal_bbx
from RenderAndCompare.datasets import NoIndent
from RenderAndCompare.geometry import alpha_to_azimuth

def main():
    root_dir_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', 'trainval.txt')
    all_types = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", default=root_dir_default, help="Path to KITTI Object directory")
    parser.add_argument("-s", "--split_file", default=splits_file_default, help="Path to split file")
    parser.add_argument("-t", "--types", type=str, nargs='+', default=['Car'], choices=all_types, help="Object type (category)")
    args = parser.parse_args()

    print "------------- Config ------------------"
    for arg in vars(args):
        print "{} \t= {}".format(arg, getattr(args, arg))

    assert osp.exists(args.root_dir), 'KITTI Object dir "{}" does not exist'.format(args.root_dir)
    assert osp.exists(args.split_file), 'Path to split file does not exist: {}'.format(args.split_file)

    image_names = [x.rstrip() for x in open(args.split_file)]
    num_of_images = len(image_names)
    print 'Using Split {} with {} images'.format(osp.basename(args.split_file), num_of_images)

    root_dir = osp.join(args.root_dir, 'training')
    label_dir = osp.join(root_dir, 'label_2')
    image_dir = osp.join(root_dir, 'image_2')
    calib_dir = osp.join(root_dir, 'calib')

    assert osp.exists(root_dir)
    assert osp.exists(label_dir)
    assert osp.exists(image_dir)
    assert osp.exists(calib_dir)

    dataset_name = 'kitti_' + osp.splitext(osp.basename(args.split_file))[0]
    dataset = Dataset(dataset_name)
    dataset.set_rootdir(root_dir)

    # Using a slight harder settings thank standard kitti hardness
    min_height = 22  # minimum height for evaluated groundtruth/detections
    max_occlusion = 2  # maximum occlusion level of the groundtruth used for evaluation
    max_truncation = 0.6  # maximum truncation level of the groundtruth used for evaluation

    print 'Generating images. May take a long time'
    for image_name in tqdm(image_names):
        image_file_path = osp.join(image_dir, image_name + '.png')
        label_file_path = osp.join(label_dir, image_name + '.txt')
        calib_file_path = osp.join(calib_dir, image_name + '.txt')

        assert osp.exists(image_file_path)
        assert osp.exists(label_file_path)
        assert osp.exists(calib_file_path)

        objects = read_kitti_object_labels(label_file_path)

        filtered_objects = []
        for obj in objects:
            if obj['type'] not in args.types:
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
        calib_data = read_kitti_calib_file(calib_file_path)
        P1 = calib_data['P1'].reshape((3, 4))
        P2 = calib_data['P2'].reshape((3, 4))

        K = P1[:3, :3]
        assert np.all(P2[:3, :3] == K)
        cam2_center = -np.linalg.inv(K).dot(P2[:, 3])
        principal_point = K[:2, 2]

        annotation = OrderedDict()
        annotation['image_file'] = osp.relpath(image_file_path, root_dir)
        annotation['image_size'] = NoIndent([W, H])
        annotation['image_intrinsics'] = NoIndent(K.astype(np.float).tolist())

        obj_infos = []
        for obj_id, obj in enumerate(filtered_objects):
            bbx_visible = np.array(obj['bbox'])
            bbx_amodal = get_kitti_amodal_bbx(obj, P2)

            azimuth = alpha_to_azimuth(obj['alpha'])
            obj_center = np.array(obj['location']) - np.array([0, obj['dimension'][0] / 2.0, 0])
            obj_center_cam2 = obj_center - cam2_center
            obj_center_cam2_xz_proj = np.array([obj_center_cam2[0], 0, obj_center_cam2[2]])
            elevation_angle = np.arctan2(np.linalg.norm(np.cross(obj_center_cam2, obj_center_cam2_xz_proj)), np.dot(obj_center_cam2, obj_center_cam2_xz_proj))
            if obj_center_cam2[1] < 0:
                elevation_angle = -elevation_angle
            elevation = math.degrees(elevation_angle) % 360
            distance = np.linalg.norm(obj_center_cam2)

            obj_info = OrderedDict()
            obj_info['category'] = obj['type']
            obj_info['viewpoint'] = NoIndent([azimuth, elevation, 0, distance])
            obj_info['bbx_visible'] = NoIndent(bbx_visible.astype(np.float).tolist())
            obj_info['bbx_amodal'] = NoIndent(bbx_amodal.astype(np.float).tolist())

            obj_infos.append(obj_info)
        annotation['objects'] = obj_infos
        dataset.add_annotation(annotation)

    print 'Finished creating dataset with {} annotations'.format(dataset.num_of_annotations())
    dataset.write_data_to_json(osp.join('temp.json'))


if __name__ == '__main__':
    main()
