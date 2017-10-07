#!/usr/bin/env python
"""
Evaluate a FastRCNN style model
"""

import os.path as osp
import re
from collections import OrderedDict
from hashlib import md5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import _init_paths
import caffe
from RenderAndCompare.datasets import Dataset, NoIndent
from RenderAndCompare.evaluation import compute_performance_metrics
from RenderAndCompare.geometry import (assert_bbx, assert_coord2D,
                                       assert_viewpoint)


def test_single_weights_file(weights_file, net, input_dataset):
    """Test already initalized net with a new set of weights"""
    net.copy_from(weights_file)
    net.layers[0].generate_datum_ids()

    num_of_images = input_dataset.num_of_annotations()
    assert net.layers[0].curr_data_ids_idx == 0
    assert net.layers[0].number_of_datapoints() == num_of_images
    assert net.layers[0].data_ids == range(num_of_images)

    assert len(net.layers[0].image_loader) == num_of_images
    assert len(net.layers[0].data_samples) == num_of_images
    assert net.layers[0].rois_per_image < 0, "rois_per_image need to be dynamic for testing"
    assert net.layers[0].imgs_per_batch == 1, "We only support one image per batch while testing"
    assert net.layers[0].flip_ratio < 0, "No flipping while testing"
    assert net.layers[0].jitter_iou_min > 1, "No jittering"

    # Create Result dataset
    result_dataset = Dataset(input_dataset.name())
    result_dataset.set_rootdir(input_dataset.rootdir())
    result_dataset.set_metainfo(input_dataset.metainfo().copy())

    # Add weight file and its md5 checksum to metainfo
    result_dataset.metainfo()['weights_file'] = weights_file
    result_dataset.metainfo()['weights_file_md5'] = md5(open(weights_file, 'rb').read()).hexdigest()

    # Set the image level fields
    for input_im_info in input_dataset.annotations():
        result_im_info = OrderedDict()
        result_im_info['image_file'] = input_im_info['image_file']
        result_im_info['image_size'] = input_im_info['image_size']
        result_im_info['image_intrinsic'] = input_im_info['image_intrinsic']
        obj_infos = []
        for input_obj_info in input_im_info['objects']:
            obj_info = OrderedDict()
            obj_info['id'] = input_obj_info['id']
            obj_info['category'] = input_obj_info['category']
            obj_info['bbx_visible'] = input_obj_info['bbx_visible']
            obj_infos.append(obj_info)
        result_im_info['objects'] = obj_infos
        assert len(result_im_info['objects']) == len(input_im_info['objects'])
        result_dataset.add_annotation(result_im_info)

    assert result_dataset.num_of_annotations() == num_of_images

    assert_funcs = {
        "viewpoint": assert_viewpoint,
        "bbx_visible": assert_bbx,
        "bbx_amodal": assert_bbx,
        "center_proj": assert_coord2D,
    }

    print 'Running inference for {} images.'.format(num_of_images)
    for image_id in tqdm.trange(num_of_images):
        # Run forward pass
        _ = net.forward()

        img_info = result_dataset.annotations()[image_id]
        expected_num_of_rois = len(img_info['objects'])
        assert net.blobs['rois'].data.shape == (expected_num_of_rois, 5)

        for info in ["bbx_amodal", "viewpoint", "center_proj"]:
            pred_info = "pred_" + info
            if pred_info in net.blobs:
                assert net.blobs[pred_info].data.shape[0] == expected_num_of_rois

        for i, obj_info in enumerate(img_info['objects']):
            for info in ["bbx_amodal", "viewpoint", "center_proj"]:
                pred_info = "pred_" + info
                if pred_info in net.blobs:
                    prediction = np.squeeze(net.blobs[pred_info].data[i, ...])
                    assert_funcs[info](prediction)
                    obj_info[info] = prediction.tolist()

    print "Evaluating results ... "
    perf_metrics_df = compute_performance_metrics(input_dataset, result_dataset)
    perf_metrics_summary_df = perf_metrics_df.describe()
    print perf_metrics_summary_df

    return result_dataset, perf_metrics_summary_df


def prepare_dataset_for_saving(dataset):
    for im_info in dataset.annotations():
        for im_info_field in ['image_size', 'image_intrinsic']:
            if im_info_field in im_info:
                im_info[im_info_field] = NoIndent(im_info[im_info_field])

        for obj_info in im_info['objects']:
            for obj_info_field in ['bbx_visible', 'bbx_amodal', 'viewpoint', 'center_proj']:
                if obj_info_field in obj_info:
                    obj_info[obj_info_field] = NoIndent(obj_info[obj_info_field])


def test_all_weights_files(weights_files, net_file, input_dataset, gpu_id):
    """Run inference on all weight files"""
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # Initialize Net
    net = caffe.Net(net_file, caffe.TEST)

    # Add dataset to datalayer
    net.layers[0].add_dataset(input_dataset)

    # print data layer params
    net.layers[0].print_params()

    perf_metrics_summaries = []
    iter_regex = re.compile('iter_([0-9]*).caffemodel')

    for weights_file in weights_files:
        weight_name = osp.splitext(osp.basename(weights_file))[0]
        print 'Working with weights_file: {}'.format(weight_name)
        result_dataset, perf_metrics_summary_df = test_single_weights_file(weights_file, net, input_dataset)

        perf_metrics_summary = {}
        perf_metrics_summary['iter'] = int(iter_regex.findall(weights_file)[0])
        for metric in perf_metrics_summary_df:
            perf_metrics_summary[metric + '_mean'] = perf_metrics_summary_df[metric]['mean']
            perf_metrics_summary[metric + '_std'] = perf_metrics_summary_df[metric]['std']

        perf_metrics_summaries.append(perf_metrics_summary)

        result_name = "{}_{}_result".format(result_dataset.name(), weight_name)
        result_dataset.set_name(result_name)
        prepare_dataset_for_saving(result_dataset)
        result_dataset.write_data_to_json(result_name + ".json")
        print '--------------------------------------------------'

    perf_metrics_df = pd.DataFrame(perf_metrics_summaries).set_index('iter')

    print 'Saving performance metrics to {}.csv'.format(input_dataset.name() + '_all_metrics')
    perf_metrics_df.to_csv(input_dataset.name() + '_all_metrics' + '.csv')

    iters = perf_metrics_df.index.values
    metric_names = []
    for metric in list(perf_metrics_df.columns.values):
        if metric.endswith('_mean'):
            metric_names.append(metric[:-5])

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(metric_names)))

    for (metric, color) in zip(metric_names, colors):
        mean = perf_metrics_df[metric + '_mean'].values
        std = perf_metrics_df[metric + '_std'].values
        plt.plot(iters, mean, label=metric, c=color)
        plt.fill_between(iters, mean - std, mean + std, facecolor=color, alpha=0.5)
    plt.legend()
    plt.xlabel('iterations')
    print 'Saving error plot to {}_all_metrics.png'.format(input_dataset.name())
    plt.savefig(input_dataset.name() + '_all_metrics.png')
    plt.show()


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Test on dataset")

    parser.add_argument("-w", "--weights_files", nargs='+', required=True, help="trained weights")
    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-d", "--dataset_file", required=True, help="Path to RenderAndCompare JSON dataset file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    args = parser.parse_args()

    assert osp.exists(args.net_file), "Net filepath {} does not exist.".format(args.net_file)
    assert osp.exists(args.dataset_file), "Dataset filepath {} does not exist.".format(args.dataset_file)
    assert args.weights_files, "Weights files cannot be empty"

    for weights_file in args.weights_files:
        assert weights_file.endswith('.caffemodel'), "Weights file {} is nopt a valid Caffe weight file".format(weights_file)

    # sort the weight files
    args.weights_files.sort(key=lambda f: int(filter(str.isdigit, f)))

    print 'Loading dataset from {}'.format(args.dataset_file)
    dataset = Dataset.from_json(args.dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    print 'User provided {} weight files'.format(len(args.weights_files))
    test_all_weights_files(args.weights_files, args.net_file, dataset, args.gpu)


if __name__ == '__main__':
    main()
