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


def compute_bbox_iou(bbxA, bbxB):
    # overlapping bounds
    x1 = max(bbxA[0], bbxB[0])
    y1 = max(bbxA[1], bbxB[1])
    x2 = min(bbxA[2], bbxB[2])
    y2 = min(bbxA[3], bbxB[3])

    # compute width and height of overlapping area
    w = x2 - x1
    h = y2 - y1

    # set invalid entries to 0 overlap
    if w <= 0 or h <= 0:
        return 0

    # get overlapping areas
    inter = w * h
    a_area = (bbxA[2] - bbxA[0]) * (bbxA[3] - bbxA[1])
    b_area = (bbxB[2] - bbxB[0]) * (bbxB[3] - bbxB[1])

    return inter / (a_area + b_area - inter)


def crop_image(image, bbox):
    W = image.shape[1]
    H = image.shape[0]

    bbx_min = np.maximum.reduce([np.array([0, 0]), np.floor(bbox[:2]).astype(int)])
    bbx_max = np.minimum.reduce([np.array([W - 1, H - 1]), np.floor(bbox[2:]).astype(int)]) + 1
    assert np.all((bbx_max - bbx_min) > 0)

    cropped_image = image[bbx_min[1]:bbx_max[1], bbx_min[0]:bbx_max[0], :]
    assert cropped_image.ndim == 3

    crop_bbx = np.array([bbx_min[0], bbx_min[1], bbx_max[0], bbx_max[1]], dtype=np.float)
    assert cropped_image.shape[0] == crop_bbx[3] - crop_bbx[1]
    assert cropped_image.shape[1] == crop_bbx[2] - crop_bbx[0]

    return cropped_image, crop_bbx


def crop_image_with_jitter(image, bbox, min_iou=0.6):
    assert 0 < min_iou < 1, 'Bad min_iou value'
    W = image.shape[1]
    H = image.shape[0]

    bbx_min = np.maximum.reduce([np.array([0, 0]), np.floor(bbox[:2]).astype(int)])
    bbx_max = np.minimum.reduce([np.array([W - 1, H - 1]), np.floor(bbox[2:]).astype(int)]) + 1
    assert np.all((bbx_max - bbx_min) > 0)
    bbx_actual = np.array([bbx_min[0], bbx_min[1], bbx_max[0], bbx_max[1]], dtype=np.float)

    bbox_w = bbx_actual[2] - bbx_actual[0]
    bbox_h = bbx_actual[3] - bbx_actual[1]

    b = 1 - min_iou
    a = 1 - (1 / min_iou)

    while True:
        jitter_ratio = a + (b - a) * np.random.rand(4)
        jitter_bbox = bbox + jitter_ratio * np.array([bbox_w, bbox_h, bbox_w, bbox_h])

        jitter_bbx_min = np.maximum.reduce([np.array([0, 0]), np.floor(jitter_bbox[:2]).astype(int)])
        jitter_bbx_max = np.minimum.reduce([np.array([W - 1, H - 1]), np.floor(jitter_bbox[2:]).astype(int)]) + 1
        if np.any((jitter_bbx_max - jitter_bbx_min) <= 0):
            continue
        jitter_bbox_actual = np.array([jitter_bbx_min[0], jitter_bbx_min[1], jitter_bbx_max[0], jitter_bbx_max[1]], dtype=np.float)
        if compute_bbox_iou(bbx_actual, jitter_bbox_actual) > min_iou:
            cropped_image = image[jitter_bbx_min[1]:jitter_bbx_max[1], jitter_bbx_min[0]:jitter_bbx_max[0], :]
            assert cropped_image.ndim == 3
            return cropped_image, jitter_bbox_actual


if __name__ == '__main__':
    kitti_object_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    assert osp.exists(kitti_object_dir), 'KITTI Object dir "{}" does not exist'.format(kitti_object_dir)

    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', 'trainval.txt')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split_file", default=splits_file_default, help="Path to split file")
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder to save the cropped images")
    parser.add_argument("-a", "--augment", type=int, default=1, choices=xrange(1, 10), help="Number of augmentations per gt box minimum 1")
    parser.add_argument("-j", "--min_jitter_iou", type=float, default=0.6, help="Minimum jitter IoU when augmenting")
    parser.add_argument("-f", "--flip_ratio", type=float, default=0, help="Ratio of samples for which we create a flipped version")
    args = parser.parse_args()

    print "------------- Config ------------------"
    for arg in vars(args):
        print "{} \t= {}".format(arg, getattr(args, arg))

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

        for obj_id, obj in enumerate(filtered_objects):
            crop_bbx_gt = np.array(obj['bbox'])
            amodal_bbx_gt = rac.datasets.get_kitti_amodal_bbx(obj, P2)

            azimuth = rac.geometry.alpha_to_azimuth(obj['alpha'])
            obj_center = np.array(obj['location']) - np.array([0, obj['dimension'][0] / 2.0, 0])
            obj_center_cam2 = obj_center - cam2_center
            obj_center_cam2_xz_proj = np.array([obj_center_cam2[0], 0, obj_center_cam2[2]])
            elevation_angle = np.arctan2(np.linalg.norm(np.cross(obj_center_cam2, obj_center_cam2_xz_proj)), np.dot(obj_center_cam2, obj_center_cam2_xz_proj))
            if obj_center_cam2[1] < 0:
                elevation_angle = -elevation_angle
            elevation = math.degrees(elevation_angle) % 360
            distance = np.linalg.norm(obj_center_cam2)

            for aug_id in xrange(args.augment):
                if aug_id == 0:
                    cropped_image, crop_bbx = crop_image(image, crop_bbx_gt)
                else:
                    cropped_image, crop_bbx = crop_image_with_jitter(image, crop_bbx_gt, args.min_jitter_iou)

                # Save cropped image with some filename
                cropped_image_name = '{}_{:04d}_aug{:02d}.png'.format(image_name, obj_id, aug_id)
                cv2.imwrite(osp.join(args.output_folder, cropped_image_name), cropped_image)

                # subtract principal point
                offset = [principal_point[0], principal_point[1], principal_point[0], principal_point[1]]
                crop_bbx_final = crop_bbx - offset
                amodal_bbx_final = amodal_bbx_gt - offset

                annotation = OrderedDict()
                annotation['image_file'] = cropped_image_name
                annotation['viewpoint'] = [azimuth, elevation, 0, distance]
                annotation['bbx_amodal'] = [amodal_bbx_final[0], amodal_bbx_final[1], amodal_bbx_final[2] - amodal_bbx_final[0], amodal_bbx_final[3] - amodal_bbx_final[1]]
                annotation['bbx_crop'] = [crop_bbx_final[0], crop_bbx_final[1], crop_bbx_final[2] - crop_bbx_final[0], crop_bbx_final[3] - crop_bbx_final[1]]
                dataset.add_annotation(annotation)

                if np.random.random() < args.flip_ratio:
                    cropped_image = np.fliplr(cropped_image)

                    # Save cropped image with some filename
                    cropped_image_name = '{}_{:04d}_aug{:02d}_flipped.png'.format(image_name, obj_id, aug_id)
                    cv2.imwrite(osp.join(args.output_folder, cropped_image_name), cropped_image)

                    crop_bbx_final = np.array([-crop_bbx_final[2], crop_bbx_final[1], -crop_bbx_final[0], crop_bbx_final[3]])
                    amodal_bbx_final = np.array([-amodal_bbx_final[2], amodal_bbx_final[1], -amodal_bbx_final[0], amodal_bbx_final[3]])

                    annotation = OrderedDict()
                    annotation['image_file'] = cropped_image_name
                    annotation['viewpoint'] = [(360.0 - azimuth) % 360, elevation, 0, distance]
                    annotation['bbx_amodal'] = [amodal_bbx_final[0], amodal_bbx_final[1], amodal_bbx_final[2] - amodal_bbx_final[0], amodal_bbx_final[3] - amodal_bbx_final[1]]
                    annotation['bbx_crop'] = [crop_bbx_final[0], crop_bbx_final[1], crop_bbx_final[2] - crop_bbx_final[0], crop_bbx_final[3] - crop_bbx_final[1]]
                    dataset.add_annotation(annotation)

    print 'Finished creating dataset with {} annotations'.format(dataset.num_of_annotations())
    dataset.write_data_to_json(osp.join(osp.dirname(args.output_folder), osp.basename(args.output_folder) + '.json'))
