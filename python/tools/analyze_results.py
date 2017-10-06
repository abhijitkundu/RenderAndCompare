#!/usr/bin/env python

"""
Analyze results against groundtruth (using dataset json files)
"""

import os.path as osp
import argparse
import matplotlib.pyplot as plt
import pandas as pd

import _init_paths
from RenderAndCompare.datasets import Dataset
from RenderAndCompare.evaluation import compute_performance_metrics, get_bbx_sizes


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

    print "perf_metrics = ...\n", perf_metrics_df.describe()

    pm_out_name = "{}.xlsx".format(pred_dataset.name())
    print 'Saving performance metrics to {}'.format(pm_out_name)
    perf_metrics_df.to_excel(pm_out_name)

    perf_metrics_df.hist(bins=96)
    plt.show()



if __name__ == '__main__':
    main()
