#!/usr/bin/env python

import os.path as osp
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
from pprint import pprint


def draw_bbx2d(image, bbx, color=(0, 255, 0)):
    cv2.rectangle(image,
                  tuple(np.floor(bbx[:2]).astype(int)),
                  tuple(np.ceil(bbx[2:]).astype(int)),
                  color, 1)


def main():
    pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1'
    sub_dataset = 'imagenet'
    category = 'car'

    annotation_dir = osp.join(pascal_root_dir, 'AnnotationsFixed', '{}_{}'.format(category, sub_dataset))
    image_dir = osp.join(pascal_root_dir, 'Images', '{}_{}'.format(category, sub_dataset))

    print "Working on annotation_dir {}".format(annotation_dir)

    assert osp.exists(annotation_dir)
    assert osp.exists(image_dir)

    anno_mat_files = glob(osp.join(annotation_dir, "*.mat"))
    print len(anno_mat_files)

    img_infos = []

    for anno_mat_file in anno_mat_files:
        record = sio.loadmat(anno_mat_file)['record'].flatten()[0]
        img_info = {}
        img_info['image_file'] = record['filename'][0]
        img_shape = np.squeeze(record['imgsize'])
        img_info['image_size'] = [img_shape[1], img_shape[0]]

        objects = record['object_infos'].flatten()
        obj_infos = []
        for obj_id in xrange(len(objects)):
            obj = objects[obj_id]
            category = obj['class'].flatten()[0]
            if category != 'car':
                continue

            occluded = obj['occluded'].flatten()[0]
            truncated = obj['truncated'].flatten()[0]
            difficult = obj['difficult'].flatten()[0]

            assert occluded == 0 or occluded == 1
            assert truncated == 0 or truncated == 1
            assert difficult == 0 or difficult == 1


            bbx = obj['bbox'].flatten()

            obj_info = {}

            obj_info['occluded'] = bool(occluded)
            obj_info['truncated'] = bool(truncated)
            obj_info['difficult'] = bool(difficult)

            obj_info['category'] = category
            obj_info['bbx_visible'] = bbx

            obj_infos.append(obj_info)
        img_info['object_infos'] = obj_infos
        img_infos.append(img_info)

    num_of_objects = sum([len(img_info['object_infos']) for img_info in img_infos])
    num_of_occuled_objects = sum([len([obj_info for obj_info in img_info['object_infos'] if obj_info['occluded']]) for img_info in img_infos])
    num_of_truncated_objects = sum([len([obj_info for obj_info in img_info['object_infos'] if obj_info['truncated']]) for img_info in img_infos])
    num_of_difficult_objects = sum([len([obj_info for obj_info in img_info['object_infos'] if obj_info['difficult']]) for img_info in img_infos])
    num_of_easy_objects = sum([len([obj_info for obj_info in img_info['object_infos'] if not obj_info['occluded'] and not obj_info['truncated'] and not obj_info['difficult']]) for img_info in img_infos])

    print "num_of_images = ", len(img_infos)
    print "num_of_objects = ", num_of_objects
    print "num_of_occuled_objects = ", num_of_occuled_objects
    print "num_of_truncated_objects = ", num_of_truncated_objects
    print "num_of_difficult_objects = ", num_of_difficult_objects
    print "num_of_easy_objects = ", num_of_easy_objects

    assert img_infos

    paused = True
    fwd = True
    i = 0

    while True:
        i = max(0, min(i, len(img_infos) - 1))
        img_info = img_infos[i]

        filtered_objects = [obj_info for obj_info in img_info['object_infos'] if not obj_info['occluded'] and not obj_info['truncated'] and not obj_info['difficult']]
        if filtered_objects:
            image = cv2.imread(osp.join(image_dir, img_info['image_file']))

            for obj_info in filtered_objects:
                draw_bbx2d(image, obj_info['bbx_visible'])

            cv2.imshow('image', image)
            key = cv2.waitKey(not paused)
            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key in [82, 83, 100, 119, 61, 43]:
                fwd = True
            elif key in [81, 84, 97, 115, 45]:
                fwd = False
            elif key == ord('p'):
                paused = not paused
        i = i + 1 if fwd else i - 1


if __name__ == '__main__':
    main()