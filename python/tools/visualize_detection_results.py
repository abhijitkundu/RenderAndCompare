#!/usr/bin/env python
"""
This script visualizes detection bounding boxes stored in ImageDataset json format
An optional score parameter can be passed to visualize boxes only above certain threshold
"""

import argparse
import os.path as osp

import cv2
import numpy as np

import _init_paths
from RenderAndCompare.datasets import ImageDataset
from RenderAndCompare.visualization import WaitKeyNavigator, draw_bbx


def main():
    parser = argparse.ArgumentParser(description="Visualize Results")
    parser.add_argument("pred_dataset_file", help="Path to predicted (results) JSON dataset file")
    parser.add_argument("-s", "--score_threshold", default=0.1, type=float, help="Score thresold")

    args = parser.parse_args()

    assert osp.exists(args.pred_dataset_file), "ImageDataset filepath {} does not exist.".format(args.pred_dataset_file)

    print 'Loading predited dataset from {}'.format(args.pred_dataset_file)
    pred_dataset = ImageDataset.from_json(args.pred_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(pred_dataset.name(), pred_dataset.num_of_images())
    print "score_threshold = {}".format(args.score_threshold)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 2048, 1024)

    wait_nav = WaitKeyNavigator(pred_dataset.num_of_images())
    wait_nav.print_key_map()

    quit_viz = False
    while not quit_viz:
        i = wait_nav.index
        image_info = pred_dataset.image_infos()[i]
        img_path = osp.join(pred_dataset.rootdir(), image_info['image_file'])
        image = cv2.imread(img_path)

        for obj_info in image_info['object_infos']:
            if 'bbx_visible' in  obj_info:
                if 'score' in obj_info:
                    if obj_info['score'] < args.score_threshold:
                        continue
                draw_bbx(image, obj_info['bbx_visible'])
                if 'category' in obj_info:
                    obj_text = obj_info['category']
                    tl = tuple(np.floor(obj_info['bbx_visible'][:2]).astype(int))
                    font_face = cv2.FONT_HERSHEY_PLAIN
                    font_scale = 0.8
                    thickness = 1
                    ts, baseline = cv2.getTextSize(obj_text, font_face, font_scale, thickness)
                    cv2.rectangle(
                        image, 
                        (tl[0], tl[1] + baseline),
                        (tl[0] + ts[0], tl[1] - ts[1]),
                        (0, 0, 0),
                        cv2.FILLED
                    )
                    cv2.addText(image, obj_text, tl, 'times', color=(0, 255, 0))


        cv2.displayOverlay('image', 'Image: {}'.format(osp.splitext(osp.basename(img_path))[0]))
        cv2.imshow('image', image)

        quit_viz = wait_nav.process_key()


if __name__ == '__main__':
    main()
