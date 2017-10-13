#!/usr/bin/env python
"""This script creates dataset json from pascal3d annotations"""

import math
import argparse
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import _init_paths
from RenderAndCompare.datasets import ImageDataset, NoIndent
from RenderAndCompare.geometry import (
    assert_viewpoint,
    assert_bbx,
    assert_coord2D,
    clip_bbx_by_image_size,
    wrap_to_pi
)


def main():
    """main function"""
    root_dir_default = osp.join(_init_paths.root_dir, 'data', 'pascal3D', 'Pascal3D-Dataset')
    split_choices = ['train', 'val', 'trainval', 'test']
    sub_dataset_choices = ['imagenet', 'pascal']
    category_choices = ['car', 'motorbike', 'bicycle', 'bus']

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", default=root_dir_default, help="Path to Pascal3d Object directory")
    parser.add_argument("-s", "--split", default='trainval', choices=split_choices, help="Split type")
    parser.add_argument("-d", "--sub_dataset", default='imagenet', choices=sub_dataset_choices, help="Sub dataset type")
    parser.add_argument("-c", "--category", type=str, nargs=1, default='car', choices=category_choices, help="Object type (category)")
    args = parser.parse_args()

    assert osp.exists(args.root_dir), "Directory '{}' do not exist".format(args.root_dir)
    anno_dir = osp.join(args.root_dir, 'AnnotationsFixed', '{}_{}'.format(args.category, args.sub_dataset))
    image_dir = osp.join(args.root_dir, 'Images', '{}_{}'.format(args.category, args.sub_dataset))
    assert osp.exists(anno_dir), "Directory '{}' do not exist".format(anno_dir)
    assert osp.exists(image_dir), "Directory '{}' do not exist".format(image_dir)

    split_file = osp.join(_init_paths.root_dir, 'data', 'pascal3D', 'splits', '{}_{}.txt'.format(args.sub_dataset, args.split))
    assert osp.exists(split_file), "Split file '{}' do not exist".format(split_file)

    print "split = {}".format(args.split)
    print "sub_dataset = {}".format(args.sub_dataset)
    print "category = {}".format(args.category)
    print "anno_dir = {}".format(anno_dir)
    print "image_dir = {}".format(image_dir)

    image_names = [x.rstrip() for x in open(split_file)]
    num_of_images = len(image_names)
    print 'Using split {} with {} images'.format(args.split, num_of_images)

    # imagenet uses JPEG while pascal images are in jpg format
    image_ext = '.JPEG' if args.sub_dataset == 'imagenet' else '.jpg'

    dataset_name = 'pascal3d_{}_{}_{}'.format(args.sub_dataset, args.split, args.category)
    dataset = ImageDataset(dataset_name)
    dataset.set_rootdir(args.root_dir)

    print "Importing dataset ..."
    for image_name in tqdm(image_names):
        anno_file = osp.join(anno_dir, image_name + '.mat')
        image_file = osp.join(image_dir, image_name + image_ext)

        if not osp.exists(anno_file):
            continue
        assert osp.exists(image_file), "Image file '{}' do not exist".format(image_file)

        image_info = OrderedDict()
        image_info['image_file'] = osp.relpath(image_file, args.root_dir)

        image = cv2.imread(image_file)
        assert image.size, "image loaded from '{}' is empty".format(image_file)

        W = image.shape[1]
        H = image.shape[0]
        image_info['image_size'] = NoIndent([W, H])

        record = sio.loadmat(anno_file)['record'].flatten()[0]
        assert record['filename'][0] == image_name + image_ext, "{} vs {}".format(record['filename'][0], image_name + image_ext)

        record_objects = record['objects'].flatten()
        obj_infos = []

        for obj_id in xrange(len(record_objects)):
            rec_obj = record_objects[obj_id]
            category = rec_obj['class'].flatten()[0]
            if category != args.category:
                continue

            rec_vp = rec_obj['viewpoint'].flatten()[0]
            distance = rec_vp['distance'].flatten()[0]
            if distance == 0.0:
                continue

            azimuth = math.radians(rec_vp['azimuth'][0, 0])
            elevation = math.radians(rec_vp['elevation'][0, 0])
            tilt = math.radians(rec_vp['theta'][0, 0])
            if azimuth == 0.0 and elevation == 0.0 and tilt == 0.0:
                continue

            viewpoint = np.array([wrap_to_pi(azimuth), wrap_to_pi(elevation), wrap_to_pi(tilt)], dtype=np.float)
            assert_viewpoint(viewpoint)

            assert rec_vp['focal'][0, 0] == 1, "rec_vp['focal'] is expected to be 1 but got {}".format(rec_vp['focal'][0, 0])
            center_proj = np.array([rec_vp['px'][0, 0], rec_vp['py'][0, 0]], dtype=np.float)
            assert_coord2D(center_proj)

            obj_info = OrderedDict()
            obj_info['category'] = category
            obj_info['occluded'] = bool(rec_obj['occluded'].flatten()[0])
            obj_info['truncated'] = bool(rec_obj['truncated'].flatten()[0])
            obj_info['difficult'] = bool(rec_obj['difficult'].flatten()[0])

            vbbx = clip_bbx_by_image_size(rec_obj['bbox'].flatten(), W, H)
            obj_info['bbx_visible'] = NoIndent(vbbx.tolist())

            if 'abbx' in rec_obj.dtype.names:
                abbx = rec_obj['abbx'].flatten()
                if abbx.shape == (4,):
                    assert_bbx(abbx)
                    obj_info['bbx_amodal'] = NoIndent(abbx.tolist())

            obj_info['viewpoint'] = NoIndent(viewpoint.tolist())
            obj_info['center_proj'] = NoIndent(center_proj.tolist())

            obj_infos.append(obj_info)

        image_info['object_infos'] = obj_infos
        dataset.add_image_info(image_info)

    total_num_of_objects = sum([len(img_info['object_infos']) for img_info in dataset.image_infos()])
    print 'Finished creating dataset with {} images and {} objects.'.format(dataset.num_of_images(), total_num_of_objects)

    num_of_objects_with_abbx = sum([len([obj_info for obj_info in img_info['object_infos'] if 'bbx_amodal' in obj_info]) for img_info in dataset.image_infos()])
    print "Number of objects with bbx_amodal information = {}".format(num_of_objects_with_abbx)
    metainfo = OrderedDict()
    metainfo['total_num_of_objects'] = total_num_of_objects
    metainfo['categories'] = NoIndent([args.category])
    dataset.set_metainfo(metainfo)

    out_json_filename = dataset_name + '.json'
    dataset.write_data_to_json(out_json_filename)


if __name__ == '__main__':
    main()
