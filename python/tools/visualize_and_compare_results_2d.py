#!/usr/bin/env python
"""
Visualize dataset 2D entities
"""

import os.path as osp
import argparse
import numpy as np
import cv2

import _init_paths
from RenderAndCompare.datasets import ImageDataset


def load_datasets(gt_dataset_file, pred_dataset_file):
    """Load gt and results datasets"""
    assert osp.exists(pred_dataset_file), "ImageDataset filepath {} does not exist.".format(pred_dataset_file)
    print 'Loading predited dataset from {}'.format(pred_dataset_file)
    pred_dataset = ImageDataset.from_json(pred_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(pred_dataset.name(), pred_dataset.num_of_images())

    gt_dataset = None
    if gt_dataset_file:
        assert osp.exists(gt_dataset_file), "ImageDataset filepath {} does not exist..".format(gt_dataset_file)
        print 'Loading groundtruth dataset from {}'.format(gt_dataset_file)
        gt_dataset = ImageDataset.from_json(gt_dataset_file)
        print 'Loaded {} dataset with {} annotations'.format(gt_dataset.name(), gt_dataset.num_of_images())
        assert gt_dataset.num_of_images() == pred_dataset.num_of_images()
        num_of_objects_gt = sum([len(image_info['object_infos']) for image_info in gt_dataset.image_infos()])
        num_of_objects_pred = sum([len(image_info['object_infos']) for image_info in gt_dataset.image_infos()])
        assert num_of_objects_gt == num_of_objects_pred, "{} ! {}".format(num_of_objects_gt, num_of_objects_pred)

    return gt_dataset, pred_dataset


def draw_object(image, obj_info, color=(0, 255, 0)):
    """draw 2d object properties"""
    bbx_amodal = np.array(obj_info['bbx_amodal'], dtype=np.float32)
    cv2.rectangle(image, tuple(bbx_amodal[:2]), tuple(bbx_amodal[2:]), color, 1)

    center_proj = np.array(obj_info['center_proj'], dtype=np.float32)
    cv2.circle(image, tuple(center_proj), 2, color, -1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze Results")
    parser.add_argument("-p", "--pred_dataset_file", required=True, help="Path to predicted (results) RenderAndCompare JSON dataset file")
    parser.add_argument("-g", "--gt_dataset_file", required=True, help="Path to groundtruth RenderAndCompare JSON dataset file")

    args = parser.parse_args()

    gt_dataset, pred_dataset = load_datasets(args.gt_dataset_file, args.pred_dataset_file)
    print "gt_dataset = {}".format(gt_dataset)
    print "pred_dataset = {}".format(pred_dataset)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    paused = True
    fwd = True

    i = 0
    while True:
        i = max(0, min(i, pred_dataset.num_of_images() - 1))
        pred_image_info = pred_dataset.image_infos()[i]
        gt_image_info = gt_dataset.image_infos()[i] if gt_dataset else None

        if gt_image_info:
            assert gt_image_info['image_file'] == pred_image_info['image_file']
            assert gt_image_info['image_size'] == pred_image_info['image_size']
            assert gt_image_info['image_intrinsic'] == pred_image_info['image_intrinsic']

        img_path = osp.join(pred_dataset.rootdir(), pred_image_info['image_file'])
        assert osp.exists(img_path), 'Image file {} does not exist'.format(img_path)
        image = cv2.imread(img_path)

        for j in xrange(len(pred_image_info['object_infos'])):
            pred_obj = pred_image_info['object_infos'][j]
            gt_obj = gt_image_info['object_infos'][j] if gt_image_info else None

            draw_object(image, pred_obj, (0, 0, 255))

            obj_text = "{}_{}".format(pred_obj['category'], pred_obj['id'])
            bbx_visible = np.array(pred_obj['bbx_visible'], dtype=np.float32)
            tl = tuple(np.floor(bbx_visible[:2]).astype(int))
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

            if gt_obj:
                assert gt_obj['id'] == pred_obj['id']
                assert gt_obj['category'] == pred_obj['category']
                draw_object(image, gt_obj, (0, 255, 0))

                cv2.line(image,
                         tuple(np.array(pred_obj['center_proj'], dtype=np.float32)),
                         tuple(np.array(gt_obj['center_proj'], dtype=np.float32)),
                         (0, 0, 255), 1, cv2.LINE_AA)

        cv2.displayOverlay('image', 'Image: {}'.format(osp.splitext(osp.basename(img_path))[0]))
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
        i = i+1 if fwd else i-1


if __name__ == '__main__':
    main()
