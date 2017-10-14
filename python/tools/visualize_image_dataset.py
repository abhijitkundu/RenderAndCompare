#!/usr/bin/env python

import argparse
import colorsys
import math
import os.path as osp
import random

import cv2
import numpy as np

import _init_paths
from RenderAndCompare.datasets import ImageDataset
from RenderAndCompare.visualization import WaitKeyNavigator, draw_bbx2d
from RenderAndCompare.geometry import (
    Pose,
    project_point,
    rotation_from_viewpoint,
    rotation_from_two_vectors,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize Image dataset")
    parser.add_argument("image_dataset_file", help="Path to ImageDataset JSON file")

    args = parser.parse_args()

    print 'Loading image dataset from {} ...'.format(args.image_dataset_file),
    dataset = ImageDataset.from_json(args.image_dataset_file)
    print 'Done.'
    print dataset

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Image', 2048, 1024)

    wait_nav = WaitKeyNavigator(dataset.num_of_images())
    wait_nav.print_key_map()

    quit_viz = False
    while not quit_viz:
        image_id = wait_nav.index
        image_info = dataset.image_infos()[image_id]

        W, H = image_info['image_size']

        img_path = osp.join(dataset.rootdir(), image_info['image_file'])
        assert osp.exists(img_path)
        image = cv2.imread(img_path)
        assert image.shape == (H, W, 3)

        if 'image_intrinsic' in image_info:
            K = np.array(image_info['image_intrinsic'], dtype=np.float)
        else:
            # Assume focal length f = 1.
            f = 200.
            K = np.array([[f, 0., W / 2.], [0., f, H / 2.], [0., 0., 1.]])

        K_inv = np.linalg.inv(K)

        for obj_info in image_info['object_infos']:
            h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
            color = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
            if 'bbx_visible' in obj_info:
                draw_bbx2d(image, obj_info['bbx_visible'], color=color, thickness=1)
            if 'bbx_amodal' in obj_info:
                draw_bbx2d(image, obj_info['bbx_amodal'], color=color, thickness=1)
            if 'center_proj'in obj_info:
                center_proj = np.array(obj_info['center_proj'], dtype=np.float)
                cv2.circle(image, tuple(center_proj.astype(np.float32)), 3, color, -1)

                if 'viewpoint' in obj_info:
                    vp = np.array(obj_info['viewpoint'], dtype=np.float)
                    R_vp = rotation_from_viewpoint(vp)
                    # R_vp = rotation_from_viewpoint(np.array([math.radians(45.), math.radians(45.), math.radians(10.)]))
                    obj_pose = Pose(R=R_vp, t=np.array([0., 0., 10.]))

                    center_proj_ray = K_inv.dot(np.append(center_proj, 1))
                    delta_rot = rotation_from_two_vectors(np.array([0., 0., 1.]), center_proj_ray)

                    obj_pose.R = delta_rot.dot(obj_pose.R)
                    obj_pose.t = delta_rot.dot(obj_pose.t)

                    obj_center_proj = project_point(K, obj_pose * (np.array([0., 0., 0.]))).astype(np.float32)
                    obj_x_proj = project_point(K, obj_pose * np.array([1., 0., 0.])).astype(np.float32)
                    obj_y_proj = project_point(K, obj_pose * np.array([0., 1., 0.])).astype(np.float32)
                    obj_z_proj = project_point(K, obj_pose * np.array([0., 0., 1.])).astype(np.float32)

                    cv2.line(image, tuple(obj_center_proj), tuple(obj_x_proj), (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(obj_center_proj), tuple(obj_y_proj), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(obj_center_proj), tuple(obj_z_proj), (255, 0, 0), 2, cv2.LINE_AA)

        cv2.displayOverlay('Image', 'Image: {}'.format(osp.splitext(osp.basename(img_path))[0]))
        cv2.imshow('Image', image)

        quit_viz = wait_nav.process_key()


if __name__ == '__main__':
    main()
