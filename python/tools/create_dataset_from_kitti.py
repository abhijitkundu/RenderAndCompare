#!/usr/bin/env python
"""This script creates dataset json from kitti object data"""

import argparse
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
from tqdm import tqdm

import _init_paths
from RenderAndCompare.datasets import (
    ImageDataset,
    NoIndent,
    get_kitti_alpha_from_object_pose,
    get_kitti_amodal_bbx,
    get_kitti_cam0_to_velo,
    get_kitti_object_pose,
    get_kitti_velo_to_cam,
    read_kitti_calib_file,
    read_kitti_object_labels
)
from RenderAndCompare.geometry import (
    Pose, project_point,
    rotation_from_two_vectors,
    rotation_from_viewpoint,
    viewpoint_from_rotation, wrap_to_pi
)


def main():
    """main function"""
    root_dir_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', 'trainval.txt')
    all_categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", default=root_dir_default, help="Path to KITTI Object directory")
    parser.add_argument("-s", "--split_file", default=splits_file_default, help="Path to split file")
    parser.add_argument("-c", "--categories", type=str, nargs='+', default=['Car'], choices=all_categories, help="Object type (category)")
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
    label_dir = osp.join(root_dir, 'label_2_updated')
    image_dir = osp.join(root_dir, 'image_2')
    calib_dir = osp.join(root_dir, 'calib')

    assert osp.exists(root_dir)
    assert osp.exists(label_dir)
    assert osp.exists(image_dir)
    assert osp.exists(calib_dir)

    dataset_name = 'kitti_' + osp.splitext(osp.basename(args.split_file))[0]
    dataset = ImageDataset(dataset_name)
    dataset.set_rootdir(root_dir)

    # Using a slight harder settings thank standard kitti hardness
    min_height = 20  # minimum height for evaluated groundtruth/detections
    max_occlusion = 2  # maximum occlusion level of the groundtruth used for evaluation
    max_truncation = 0.7  # maximum truncation level of the groundtruth used for evaluation

    total_num_of_objects = 0

    print 'Creating ImageDataset. May take long time'
    for image_name in tqdm(image_names):
        image_file_path = osp.join(image_dir, image_name + '.png')
        label_file_path = osp.join(label_dir, image_name + '.txt')
        calib_file_path = osp.join(calib_dir, image_name + '.txt')

        assert osp.exists(image_file_path)
        assert osp.exists(label_file_path)
        assert osp.exists(calib_file_path)

        objects = read_kitti_object_labels(label_file_path)

        # filter the objects based on kitti hardness criteria
        filtered_objects = {}
        for obj_id, obj in enumerate(objects):
            if obj['type'] not in args.categories:
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
                filtered_objects[obj_id] = obj

        if not filtered_objects:
            continue

        total_num_of_objects += len(filtered_objects)

        image = cv2.imread(image_file_path)
        W = image.shape[1]
        H = image.shape[0]
        calib_data = read_kitti_calib_file(calib_file_path)
        P0 = calib_data['P0'].reshape((3, 4))
        P2 = calib_data['P2'].reshape((3, 4))
        K = P0[:3, :3]
        assert np.all(P2[:3, :3] == K)

        cam2_center = -np.linalg.inv(K).dot(P2[:, 3])

        velo_T_cam0 = get_kitti_cam0_to_velo(calib_data)
        velo_T_cam2 = velo_T_cam0 * Pose(t=cam2_center)
        cam2_T_velo = get_kitti_velo_to_cam(calib_data, cam2_center)
        assert np.allclose(velo_T_cam2.inverse().matrix(), cam2_T_velo.matrix())

        annotation = OrderedDict()
        annotation['image_file'] = osp.relpath(image_file_path, root_dir)
        annotation['image_size'] = NoIndent([W, H])
        annotation['image_intrinsic'] = NoIndent(K.astype(np.float).tolist())

        obj_infos = []
        for obj_id in sorted(filtered_objects):
            obj = filtered_objects[obj_id]
            obj_pose_cam2 = get_kitti_object_pose(obj, velo_T_cam0, cam2_center)
            obj_pose_cam0 = get_kitti_object_pose(obj, velo_T_cam0, np.zeros(3))
            assert np.allclose(obj_pose_cam0.t - obj_pose_cam2.t, cam2_center)

            bbx_visible = np.array(obj['bbox'])
            bbx_amodal = get_kitti_amodal_bbx(obj, K, obj_pose_cam2)

            obj_origin_proj = project_point(K, obj_pose_cam2.t)
            distance = np.linalg.norm(obj_pose_cam2.t)

            delta_rot = rotation_from_two_vectors(obj_pose_cam2.t, np.array([0., 0., 1.]))
            obj_rel_rot = np.matmul(delta_rot, obj_pose_cam2.R)
            assert np.allclose(delta_rot.dot(obj_pose_cam2.t), np.array([0., 0., distance]))

            viewpoint = viewpoint_from_rotation(obj_rel_rot)

            R_vp = rotation_from_viewpoint(viewpoint)
            assert np.allclose(R_vp, obj_rel_rot, rtol=1e-03), "R_vp = \n{}\nobj_rel_rot = \n{}\n".format(R_vp, obj_rel_rot)
            assert np.allclose(np.matmul(delta_rot.T, R_vp), obj_pose_cam2.R, rtol=1e-04)

            pred_alpha = get_kitti_alpha_from_object_pose(obj_pose_cam2, velo_T_cam2)
            alpha_diff = wrap_to_pi(pred_alpha - obj['alpha'])
            assert np.abs(alpha_diff) < 0.011, "{} vs {}. alpha_diff={}".format(pred_alpha, obj['alpha'], alpha_diff)

            obj_info = OrderedDict()

            obj_info['id'] = obj_id
            obj_info['category'] = obj['type'].lower()
            obj_info['dimension'] = NoIndent(obj['dimension'][::-1])  # [length, width, height]
            obj_info['bbx_visible'] = NoIndent(bbx_visible.tolist())
            obj_info['bbx_amodal'] = NoIndent(np.around(bbx_amodal, decimals=6).tolist())
            obj_info['viewpoint'] = NoIndent(np.around(viewpoint, decimals=6).tolist())
            obj_info['center_proj'] = NoIndent(np.around(obj_origin_proj, decimals=6).tolist())
            obj_info['center_dist'] = distance

            obj_infos.append(obj_info)
        annotation['object_infos'] = obj_infos
        dataset.add_image_info(annotation)

    print 'Finished creating dataset with {} images and {} objects.'.format(dataset.num_of_images(), total_num_of_objects)

    metainfo = OrderedDict()
    metainfo['total_num_of_objects'] = total_num_of_objects
    metainfo['categories'] = NoIndent([x.lower() for x in args.categories])
    metainfo['min_height'] = min_height
    metainfo['max_occlusion'] = max_occlusion
    metainfo['max_truncation'] = max_truncation
    dataset.set_metainfo(metainfo)

    out_json_filename = dataset_name + '.json'
    print 'Saving annotations to {}'.format(out_json_filename)
    dataset.write_data_to_json(out_json_filename)


if __name__ == '__main__':
    main()
