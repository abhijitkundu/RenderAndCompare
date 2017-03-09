#!/usr/bin/env python

import _init_paths
import cv2
import numpy as np
import os.path as osp
import RenderAndCompare as rac
from itertools import chain


if __name__ == '__main__':
    kitti_object_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    assert osp.exists(kitti_object_dir), 'KITTI Object dir "{}" does not exist'.format(kitti_object_dir)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--set", default='training', help="training or testing set")
    parser.add_argument("label_files", nargs='+', help="Path to kitti label files")
    parser.add_argument("-t", "--score_thresh", type=float, default=0.0, help="Score Threshold")
    args = parser.parse_args()

    image_dir = osp.join(kitti_object_dir, args.set, 'image_2')
    calib_dir = osp.join(kitti_object_dir, args.set, 'calib')

    assert osp.exists(image_dir)
    assert osp.exists(calib_dir)

    if args.set == 'training':
        gt_labels_dir = osp.join(kitti_object_dir, args.set, 'label_2')
        assert osp.exists(gt_labels_dir)

    print 'Computing min/max score for {} images'.format(len(args.label_files))
    # Load all results in dict
    dataset_objects = {}
    for label_file_path in args.label_files:
        image_stem = osp.splitext(osp.basename(label_file_path))[0]
        assert osp.exists(label_file_path)
        objects = rac.datasets.read_kitti_object_labels(label_file_path)
        dataset_objects[image_stem] = objects

    scores = [[obj['score'] for obj in image_objects] for image_objects in dataset_objects.values()]
    min_score = min(chain(*scores))
    max_score = max(chain(*scores))
    score_multiplier = 1.0 / (max_score - min_score)
    print 'Score Range: %.2f -- %.2f' % (min_score, max_score)

    print 'Visualizing results on {} images'.format(len(args.label_files))
    cv2.namedWindow('Results', cv2.WINDOW_AUTOSIZE)
    for label_file_path in args.label_files:
        image_stem = osp.splitext(osp.basename(label_file_path))[0]

        image_file_path = osp.join(image_dir, image_stem + '.png')
        calib_file_path = osp.join(calib_dir, image_stem + '.txt')

        assert osp.exists(image_file_path)
        assert osp.exists(label_file_path)
        assert osp.exists(calib_file_path)

        image = cv2.imread(image_file_path)

        objects = rac.datasets.read_kitti_object_labels(label_file_path)

        for obj in objects:
            bbx = np.floor(np.asarray(obj['bbox'])).astype(int)
            normalized_score = (obj['score'] - min_score) * score_multiplier

            if normalized_score < args.score_thresh:
                continue

            bbx_str = obj['type'] + ' %.2f' % normalized_score
            cv2.rectangle(image,
                          (bbx[0], bbx[1]),
                          (bbx[2], bbx[3]),
                          (0, 255, 0), 1)
            cv2.putText(image, bbx_str, (bbx[0] + 5, bbx[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Results', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
