#!/usr/bin/env python

import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
from tqdm import tqdm

import _init_paths
from RenderAndCompare.datasets import (Dataset,
                                       NoIndent,
                                       alpha_to_azimuth,
                                       get_kitti_amodal_bbx,
                                       get_kitti_object_pose,
                                       read_kitti_calib_file,
                                       read_kitti_object_labels)
from RenderAndCompare.geometry import (project_point,
                                       rotation_from_two_vectors,
                                       rotation_from_viewpoint,
                                       viewpoint_from_rotation)


def main():
    root_dir_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', 'trainval.txt')
    all_categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

    import argparse
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
    dataset = Dataset(dataset_name)
    dataset.set_rootdir(root_dir)

    # Using a slight harder settings thank standard kitti hardness
    min_height = 20  # minimum height for evaluated groundtruth/detections
    max_occlusion = 2  # maximum occlusion level of the groundtruth used for evaluation
    max_truncation = 0.7  # maximum truncation level of the groundtruth used for evaluation

    total_num_of_objects = 0

    print 'Generating images. May take long time'
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
                filtered_objects.append(obj)

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

        annotation = OrderedDict()
        annotation['image_file'] = osp.relpath(image_file_path, root_dir)
        annotation['image_size'] = NoIndent([W, H])
        annotation['image_intrinsic'] = NoIndent(K.astype(np.float).tolist())

        obj_infos = []
        for obj in filtered_objects:
            bbx_visible = np.array(obj['bbox'])
            bbx_amodal = get_kitti_amodal_bbx(obj, K, cam2_center)

            R, t = get_kitti_object_pose(obj, cam2_center)

            obj_center_cam0 = np.array(obj['location']) - np.array([0, obj['dimension'][0] / 2.0, 0])
            obj_center_cam2 = obj_center_cam0 - cam2_center

            assert np.allclose(project_point(P2, obj_center_cam0), project_point(K, t))
            assert np.allclose(np.linalg.norm(obj_center_cam2), np.linalg.norm(t))

            obj_origin_proj = project_point(K, t)
            distance = np.linalg.norm(t)

            delta_rot = rotation_from_two_vectors(t, np.array([0., 0., 1.]))
            obj_rel_rot = np.matmul(delta_rot, R)
            assert np.allclose(delta_rot.dot(t), np.array([0., 0., distance]))

            viewpoint = viewpoint_from_rotation(obj_rel_rot)

            R_vp = rotation_from_viewpoint(viewpoint)
            assert np.allclose(R_vp, obj_rel_rot), "R_vp = \n{}\nobj_rel_rot = \n{}\n".format(R_vp, obj_rel_rot)
            assert np.allclose(np.matmul(delta_rot.T, R_vp), R)

            obj_info = OrderedDict()

            obj_info['category'] = obj['type']
            obj_info['dimension'] = NoIndent(obj['dimension'][::-1])  # [length, width, height]
            obj_info['bbx_visible'] = NoIndent(bbx_visible.tolist())
            obj_info['bbx_amodal'] = NoIndent(np.around(bbx_amodal, decimals=6).tolist())
            obj_info['viewpoint'] = NoIndent(np.around(viewpoint, decimals=6).tolist())
            obj_info['center_proj'] = NoIndent(np.around(obj_origin_proj, decimals=6).tolist())
            obj_info['center_dist'] = distance

            obj_infos.append(obj_info)
        annotation['objects'] = obj_infos
        dataset.add_annotation(annotation)

    print 'Finished creating dataset with {} images and {} objects.'.format(dataset.num_of_annotations(), total_num_of_objects)

    metainfo = OrderedDict()
    metainfo['total_num_of_objects'] = total_num_of_objects
    metainfo['categories'] = NoIndent(args.categories)
    metainfo['min_height'] = min_height
    metainfo['max_occlusion'] = max_occlusion
    metainfo['max_truncation'] = max_truncation
    dataset.set_metainfo(metainfo)

    out_json_filename = dataset_name + '.json'
    print 'Saving annotations to {}'.format(out_json_filename)
    dataset.write_data_to_json(out_json_filename)


if __name__ == '__main__':
    main()
