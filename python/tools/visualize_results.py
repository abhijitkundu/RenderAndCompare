#!/usr/bin/env python

import os.path as osp
import argparse
import numpy as np
import cv2

import _init_paths
from RenderAndCompare.datasets import Dataset


def draw_bbx2d(image, bbx, color=(0, 255, 0)):
    cv2.rectangle(image,
                  tuple(np.floor(bbx[:2]).astype(int)),
                  tuple(np.ceil(bbx[2:]).astype(int)),
                  color, 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize Results")
    parser.add_argument("pred_dataset_file", help="Path to predicted (results) JSON dataset file")
    parser.add_argument("-s", "--score_threshold", default=0.1, type=float, help="Score thresold")
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")

    args = parser.parse_args()

    assert osp.exists(args.pred_dataset_file), "Dataset filepath {} does not exist.".format(args.pred_dataset_file)

    print 'Loading predited dataset from {}'.format(args.pred_dataset_file)
    pred_dataset = Dataset.from_json(args.pred_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(pred_dataset.name(), pred_dataset.num_of_annotations())

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    i = 0
    while True:
        i = max(0, min(i, pred_dataset.num_of_annotations() - 1))
        image_info = pred_dataset.annotations()[i]
        img_path = osp.join(pred_dataset.rootdir(), image_info['image_file'])
        image = cv2.imread(img_path)

        for obj_info in image_info['objects']:
            if 'bbx_visible' in  obj_info:
                if 'score' in obj_info:
                    if obj_info['score'] < args.score_threshold:
                        continue
                draw_bbx2d(image, obj_info['bbx_visible'])
                if 'category' in obj_info:
                    obj_text = obj_info['category']
                    cv2.putText(image, 
                                obj_text, 
                                tuple(np.floor(obj_info['bbx_visible'][:2]).astype(int)), 
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)


        cv2.displayOverlay('image', 'Image: {}'.format(osp.splitext(osp.basename(img_path))[0]))
        cv2.imshow('image', image)
        key = cv2.waitKey(args.pause)

        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key in [82, 83, 100, 119, 61, 43]:
            i += 1
        elif key in [81, 84, 97, 115, 45]:
            i -= 1


if __name__ == '__main__':
    main()
