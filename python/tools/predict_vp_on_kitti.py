#!/usr/bin/env python

import os.path as osp
from os import makedirs

import cv2
import numpy as np
from tqdm import tqdm

import _init_paths
import caffe
import RenderAndCompare as rac


def deprocess(image, mean_bgr):
    deprocessed_image = image.transpose(1, 2, 0)
    deprocessed_image += mean_bgr
    return np.uint8(deprocessed_image)


if __name__ == '__main__':
    import argparse
    description = ('Predict viewpoints from KITTI results files')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights_file", required=True, help="trained weights")
    parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id.")
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder to save the results")
    parser.add_argument("-s", "--set", default='training', help="training or testing set")
    parser.add_argument("label_files", nargs='+', help="Path to kitti label files")

    args = parser.parse_args()

    # initialize net
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()    

    net = caffe.Net(args.net_file, args.weights_file, caffe.TEST)

    kitti_object_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object')
    assert osp.exists(kitti_object_dir), 'KITTI Object dir "{}" does not exist'.format(kitti_object_dir)
    image_dir = osp.join(kitti_object_dir, args.set, 'image_2')
    assert osp.exists(image_dir), 'KITTI image dir "{}" does not exist'.format(image_dir)

    if not osp.exists(args.output_folder):
        print 'Created new output directory at {}'.format(args.output_folder)
        makedirs(args.output_folder)
    else:
        print 'Will overwrite results in {}'.format(args.output_folder)

    print 'Predicting on {} images'.format(len(args.label_files))

    for label_file_path in tqdm(args.label_files):
        image_stem = osp.splitext(osp.basename(label_file_path))[0]

        image_file_path = osp.join(image_dir, image_stem + '.png')
        assert osp.exists(image_file_path)
        image = cv2.imread(image_file_path).astype(np.float32)

        W = image.shape[1]
        H = image.shape[0]

        assert osp.exists(label_file_path)
        objects = rac.datasets.read_kitti_object_labels(label_file_path)

        # Get a filtered list of objects
        filtered_objects = []
        for obj in objects:
            if obj['type'] != 'Car':
                continue
            bbx = np.array(obj['bbox'])
            bbx_min = np.maximum.reduce([np.array([0, 0]), np.floor(bbx[:2]).astype(int)])
            bbx_max = np.minimum.reduce([np.array([W - 1, H - 1]), np.floor(bbx[2:]).astype(int)])
            if np.any((bbx_max - bbx_min) <= 0):
                continue
            filtered_objects.append(obj)

        num_of_objects = len(filtered_objects)
        if num_of_objects > 0:
            boxes = [np.array(obj['bbox']) for obj in filtered_objects]

            predictions = rac.prediction.get_predictions_on_image_boxes(image,
                                                                        boxes,
                                                                        net,
                                                                        ['azimuth_pred', 'elevation_pred', 'tilt_pred'],
                                                                        args.mean_bgr)

            assert predictions['azimuth_pred'].size == num_of_objects
            assert predictions['azimuth_pred'].size == predictions['azimuth_pred'].shape[0]
            azimuths = predictions['azimuth_pred'].reshape(num_of_objects)
            # elevations = np.squeeze(predictions['elevation_pred'])
            # tilts = np.squeeze(predictions['tilt_pred'])
            # viewpoints_pred = zip(azimuths, elevations, tilts)

            # Update filtered_objects.alpha
            for i, obj in enumerate(filtered_objects):
                obj['alpha'] = rac.geometry.azimuth_to_alpha(azimuths[i])

        # Save updated filtered_objects
        out_label_filepath = osp.join(args.output_folder, image_stem + '.txt')
        rac.datasets.write_kitti_object_labels(filtered_objects, out_label_filepath)
