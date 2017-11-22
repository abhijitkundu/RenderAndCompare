#!/usr/bin/env python

"""
This script splits an image dataset by category
"""


import argparse
import json
import os.path as osp
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

import _init_paths
from RenderAndCompare.datasets import ImageDataset, NoIndent
from RenderAndCompare.geometry import clip_bbx_by_image_size, assert_bbx


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dataset_file", required=True, type=str, help="Path to image dataset file to split")
    parser.add_argument("-c", "--category", required=True, type=str, help="category to separate out")
    parser.add_argument("-s", "--score_thresh", default=0.0, type=float, help="minimum score")
    args = parser.parse_args()

    assert osp.isfile(args.image_dataset_file), '{} either do not exist or not a file'.format(args.image_dataset_file)

    print('Loading image dataset from {}'.format(args.image_dataset_file))
    image_datset = ImageDataset.from_json(args.image_dataset_file)
    print(image_datset)
    num_of_objects = sum([len(img_info['object_infos']) for img_info in image_datset.image_infos()])
    print("total number of objects = {}".format(num_of_objects))

    new_image_infos = []

    print('selecting object_infos with category {}'.format(args.category))
    for im_info in tqdm(image_datset.image_infos()):
            new_im_info = OrderedDict()

            for im_info_field in ['image_file', 'segm_file']:
                if im_info_field in im_info:
                    new_im_info[im_info_field] = im_info[im_info_field]

            for im_info_field in ['image_size', 'image_intrinsic']:
                if im_info_field in im_info:
                    new_im_info[im_info_field] = NoIndent(im_info[im_info_field])
            
            W = im_info['image_size'][0]
            H = im_info['image_size'][1]

            new_obj_infos = []
            for obj_id, obj_info in enumerate(im_info['object_infos']):
                if obj_info['category'] != args.category:
                    continue

                if obj_info['score'] < args.score_thresh:
                    continue

                new_obj_info = OrderedDict()

                if 'id' not in obj_info:
                    obj_info['id'] = obj_id + 1

                vbbx = np.array(obj_info['bbx_visible'])
                assert_bbx(vbbx)
                vbbx = clip_bbx_by_image_size(vbbx, W, H)
                assert_bbx(vbbx)
                new_obj_info['bbx_visible'] = NoIndent(vbbx.tolist())

                for obj_info_field in ['id', 'category']:
                    if obj_info_field in obj_info:
                        new_obj_info[obj_info_field] = obj_info[obj_info_field]

                for obj_info_field in ['viewpoint', 'bbx_amodal', 'center_proj', 'dimension']:
                    if obj_info_field in obj_info:
                        new_obj_info[obj_info_field] = NoIndent(obj_info[obj_info_field])

                for obj_info_field in ['center_dist', 'occlusion', 'truncation', 'shape_file', 'score']:
                    if obj_info_field in obj_info:
                        new_obj_info[obj_info_field] = obj_info[obj_info_field]
                new_obj_infos.append(new_obj_info)

            if new_obj_infos:
                new_im_info['object_infos'] = new_obj_infos
                new_image_infos.append(new_im_info)

    new_dataset = ImageDataset(name="{}_{}".format(image_datset.name(), args.category))
    new_dataset.set_image_infos(new_image_infos)
    new_dataset.set_rootdir(image_datset.rootdir())
    num_of_objects = sum([len(img_info['object_infos']) for img_info in new_dataset.image_infos()])

    metainfo = OrderedDict()
    metainfo['total_num_of_objects'] = num_of_objects
    metainfo['categories'] = NoIndent([args.category])
    metainfo['score_thresh'] = args.score_thresh
    new_dataset.set_metainfo(metainfo)

    print(new_dataset)
    print("new number of objects = {}".format(num_of_objects))

    new_dataset.write_data_to_json(new_dataset.name() + ".json")


if __name__ == '__main__':
    main()
