#!/usr/bin/env python

"""
Analyze results against groundtruth (using dataset json files)
"""

import os.path as osp
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import _init_paths
from RenderAndCompare.datasets import ImageDataset
from RenderAndCompare.evaluation import compute_performance_metrics, get_bbx_sizes


def load_datasets(gt_dataset_file, pred_dataset_file):
    """Load gt and results datasets"""
    assert osp.exists(gt_dataset_file), "ImageDataset filepath {} does not exist..".format(gt_dataset_file)
    assert osp.exists(pred_dataset_file), "ImageDataset filepath {} does not exist.".format(pred_dataset_file)

    print 'Loading groundtruth dataset from {}'.format(gt_dataset_file)
    gt_dataset = ImageDataset.from_json(gt_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(gt_dataset.name(), gt_dataset.num_of_images())
    print 'Loading predited dataset from {}'.format(pred_dataset_file)
    pred_dataset = ImageDataset.from_json(pred_dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(pred_dataset.name(), pred_dataset.num_of_images())
    assert gt_dataset.num_of_images() == pred_dataset.num_of_images()

    num_of_objects_gt = sum([len(image_info['object_infos']) for image_info in gt_dataset.image_infos()])
    num_of_objects_pred = sum([len(image_info['object_infos']) for image_info in gt_dataset.image_infos()])
    assert num_of_objects_gt == num_of_objects_pred, "{} ! {}".format(num_of_objects_gt, num_of_objects_pred)

    return gt_dataset, pred_dataset


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze Results")
    parser.add_argument("-g", "--gt_dataset_file", required=True, help="Path to groundtruth RenderAndCompare JSON dataset file")
    parser.add_argument("-p", "--pred_dataset_file", required=True, help="Path to predicted (results) RenderAndCompare JSON dataset file")
    args = parser.parse_args()

    # load both the datasets
    gt_dataset, pred_dataset = load_datasets(args.gt_dataset_file, args.pred_dataset_file)

    # compute perf metrics
    perf_metrics_df = compute_performance_metrics(gt_dataset, pred_dataset)

    # get bbx sizes
    bbx_sizes_df = get_bbx_sizes(gt_dataset)

    assert (perf_metrics_df.index == bbx_sizes_df.index).all()
    perf_metrics_df = pd.concat([perf_metrics_df, bbx_sizes_df], axis=1)

    pd.set_option('display.width', 1000)
    print "perf_metrics = ...\n", perf_metrics_df.describe()

    if 'vp_geo_error_deg' in perf_metrics_df.columns:
        vp_geo_errors = perf_metrics_df['vp_geo_error_deg'].values
        num_of_objects_gt = sum([len(image_info['object_infos']) for image_info in gt_dataset.image_infos()])
        assert vp_geo_errors.shape == (num_of_objects_gt,)
        med_err_deg = np.median(vp_geo_errors)
        acc_pi_by_6 = float(np.sum(vp_geo_errors < np.degrees(np.pi / 6))) / vp_geo_errors.size
        acc_pi_by_12 = float(np.sum(vp_geo_errors < np.degrees(np.pi / 12))) / vp_geo_errors.size
        acc_pi_by_24 = float(np.sum(vp_geo_errors < np.degrees(np.pi / 24))) / vp_geo_errors.size
        print 'Geodesic med_err_deg= {}, acc_pi_by_6= {}, acc_pi_by_12= {}, acc_pi_by_24= {}'.format(med_err_deg, acc_pi_by_6, acc_pi_by_12, acc_pi_by_24)

    pm_out_name = "{}.xlsx".format(pred_dataset.name())
    print 'Saving performance metrics to {}'.format(pm_out_name)
    perf_metrics_df.to_excel(pm_out_name)

    perf_metrics_df.hist(bins=96)
    plt.show()


if __name__ == '__main__':
    main()
