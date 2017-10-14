#!/usr/bin/env python

import argparse
import os.path as osp

import cv2
import numpy as np

import _init_paths
from RenderAndCompare.datasets import ImageDataset
from RenderAndCompare.visualization import WaitKeyNavigator, draw_bbx2d


def main():
    parser = argparse.ArgumentParser(description="Visualize Image dataset")
    parser.add_argument("image_dataset_file", help="Path to ImageDataset JSON file")

    args = parser.parse_args()

    print 'Loading image dataset from {} ...'.format(args.image_dataset_file),
    dataset = ImageDataset.from_json(args.image_dataset_file)
    print 'Done.'
    print dataset

    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)

    wait_nav = WaitKeyNavigator(dataset.num_of_images())
    wait_nav.print_key_map()

    quit_viz = False
    while not quit_viz:
        image_id = wait_nav.index
        image_info = dataset.image_infos()[image_id]
        img_path = osp.join(dataset.rootdir(), image_info['image_file'])
        assert osp.exists(img_path)
        image = cv2.imread(img_path)

        cv2.displayOverlay('Image', 'Image: {}'.format(osp.splitext(osp.basename(img_path))[0]))
        cv2.imshow('Image', image)

        quit_viz = wait_nav.process_key()


if __name__ == '__main__':
    main()
