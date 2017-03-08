#!/usr/bin/env python

import _init_paths
import os.path as osp
import RenderAndCompare as rac
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    kitti_object_gt_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object', 'training', 'label_2')
    assert osp.exists(kitti_object_gt_dir), 'KITTI GT labels directory "{}" does not exist'.format(kitti_object_gt_dir)

    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', '3dvp_val.txt')

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--result_dir", required=True, help="Path to Results folder")
    parser.add_argument("-g", "--gt_dir", default=kitti_object_gt_dir, help="KITTI GT label directory")
    parser.add_argument("-s", "--split_file", default=splits_file_default, help="Path to split file")
    parser.add_argument("-d", "--display", type=int, default=1, help="Set to zero if you do not wanna display plots")

    args = parser.parse_args()

    min_height = 25  # minimum height for evaluated groundtruth/detections
    max_occlusion = 2  # maximum occlusion level of the groundtruth used for evaluation
    max_truncation = 0.5  # maximum truncation level of the groundtruth used for evaluation

    # Load ground truth
    # Load Results

    # For each valid object (to be eavaluated), find the best result bbx in terms of overlap

    best_ious = []

    image_ids = [line.strip() for line in open(args.split_file).readlines()]

    for img_id in image_ids:
        gt_objects = rac.datasets.read_kitti_object_label_file(osp.join(args.gt_dir, img_id + '.txt'))
        results_objects = rac.datasets.read_kitti_object_label_file(osp.join(args.result_dir, img_id + '.txt'))

        for gt_obj in gt_objects:
            if gt_obj['type'] != 'Car':
                continue

            gt_bbx = np.asarray(gt_obj['bbox'])
            if (gt_bbx[3] - gt_bbx[1]) < min_height:
                continue
            if gt_obj['occlusion'] > max_occlusion:
                continue
            if gt_obj['truncation'] > max_truncation:
                continue

            best_iou = 0
            for obj in results_objects:
                bbx = np.asarray(obj['bbox'])
                iou = compute_bbox_iou(gt_bbx, bbx)
                best_iou = max(best_iou, iou)

            best_ious.append(best_iou)

    print len(best_ious)
    print np.mean(best_ious)

    plt.hist(np.asarray(best_ious), bins=30)
    plt.xlabel('IOUs');
    plt.show()
