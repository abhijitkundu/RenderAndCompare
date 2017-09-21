#!/usr/bin/env python

"""
Analyze results against groundtruth (using dataset json files)
"""

import os.path as osp
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import _init_paths
from RenderAndCompare.datasets import Dataset
from RenderAndCompare.geometry import assert_viewpoint, assert_bbx


def load_datasets(gt_dataset_file, pred_dataset_file):
    """Load gt and results datasets"""
    assert osp.exists(gt_dataset_file), "Dataset filepath {} does not exist..".format(gt_dataset_file)
    assert osp.exists(pred_dataset_file), "Dataset filepath {} does not exist.".format(pred_dataset_file)

    print 'Loading groundtruth dataset from {}'.format(gt_dataset_file)
    gt_dataset = Dataset.from_json(gt_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(gt_dataset.name(), gt_dataset.num_of_annotations())
    print 'Loading predited dataset from {}'.format(pred_dataset_file)
    pred_dataset = Dataset.from_json(pred_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(pred_dataset.name(), pred_dataset.num_of_annotations())
    assert gt_dataset.num_of_annotations() == pred_dataset.num_of_annotations()

    num_of_objects_gt = sum([len(image_info['objects']) for image_info in gt_dataset.annotations()])
    num_of_objects_pred = sum([len(image_info['objects']) for image_info in gt_dataset.annotations()])
    assert num_of_objects_gt == num_of_objects_pred, "{} ! {}".format(num_of_objects_gt, num_of_objects_pred)

    return gt_dataset, pred_dataset


def compute_viewpoint_error(vpA, vpB, use_degrees=True):
    """compute viewpoint error (absolute) in degrees (default) or radians (if use_degrees=False)"""
    angle_error = (np.array(vpA) - np.array(vpB) + np.pi) % (2 * np.pi) - np.pi
    assert_viewpoint(angle_error)
    if use_degrees:
        angle_error = np.degrees(angle_error)
    return np.fabs(angle_error)


def compute_coord_error(pointA, pointB):
    """compute 2d distance (L2) bw two points"""
    return np.linalg.norm(np.array(pointA) - np.array(pointB))


def compute_bbx_iou(bbxA_, bbxB_):
    """Compute iou of two bounding boxes"""
    bbxA = np.array(bbxA_)
    bbxB = np.array(bbxB_)
    assert_bbx(bbxA)
    assert_bbx(bbxB)

    x1 = np.maximum(bbxA[0], bbxB[0])
    y1 = np.maximum(bbxA[1], bbxB[1])

    x2 = np.minimum(bbxA[2], bbxB[2])
    y2 = np.minimum(bbxA[3], bbxB[3])

    # compute width and height of overlapping area
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return 0

    inter = w * h
    a_area = (bbxA[2:] - bbxA[:2]).prod()
    b_area = (bbxB[2:] - bbxB[:2]).prod()
    iou = inter / (a_area + b_area - inter)

    return float(iou)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze Results")
    parser.add_argument("-g", "--gt_dataset_file", required=True, help="Path to groundtruth RenderAndCompare JSON dataset file")
    parser.add_argument("-p", "--pred_dataset_file", required=True, help="Path to predicted (results) RenderAndCompare JSON dataset file")
    args = parser.parse_args()

    gt_dataset, pred_dataset = load_datasets(args.gt_dataset_file, args.pred_dataset_file)

    perf_metrics = []

    print "Computing performance metrics:"
    for gt_image_info, pred_image_info in tqdm(zip(gt_dataset.annotations(), pred_dataset.annotations())):
        assert gt_image_info['image_file'] == pred_image_info['image_file']
        assert gt_image_info['image_size'] == pred_image_info['image_size']
        assert gt_image_info['image_intrinsic'] == pred_image_info['image_intrinsic']

        gt_objects = gt_image_info['objects']
        pred_objects = pred_image_info['objects']
        assert len(gt_objects) == len(pred_objects)

        for gt_obj, pred_obj in zip(gt_objects, pred_objects):
            assert gt_obj['id'] == pred_obj['id']
            assert gt_obj['category'] == pred_obj['category']

            pm = {}
            pm['obj_id'] = "{}_{}".format(osp.splitext(osp.basename(gt_image_info['image_file']))[0], gt_obj['id'])

            # compute viewpoint error
            if all('viewpoint' in obj_info for obj_info in (gt_obj, pred_obj)):
                vp_error_deg = compute_viewpoint_error(gt_obj['viewpoint'], pred_obj['viewpoint'])
                pm['error_azimuth_deg'] = vp_error_deg[0]
                pm['error_elevation_deg'] = vp_error_deg[1]
                pm['error_tilt_deg'] = vp_error_deg[2]

            # compute pixel center_proj error
            if all('center_proj' in obj_info for obj_info in (gt_obj, pred_obj)):
                pm['error_center_proj_pixels'] = compute_coord_error(gt_obj['center_proj'], pred_obj['center_proj'])

            # compute bbx overlap error
            if all('bbx_amodal' in obj_info for obj_info in (gt_obj, pred_obj)):
                pm['iou_bbx_amodal'] = compute_bbx_iou(gt_obj['bbx_amodal'], pred_obj['bbx_amodal'])

            perf_metrics.append(pm)

    perf_metrics_df = pd.DataFrame(perf_metrics).set_index('obj_id')
    print perf_metrics_df.describe()
    perf_metrics_df.hist(bins=96)
    plt.show()

    pm_out_name = "{}.xlsx".format(pred_dataset.name())
    print 'Saving performance metrics to {}'.format(pm_out_name)
    perf_metrics_df.to_excel(pm_out_name)


if __name__ == '__main__':
    main()
