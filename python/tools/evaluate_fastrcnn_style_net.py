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
from RenderAndCompare.datasets import ImageDataset, NoIndent
from RenderAndCompare.evaluation import compute_performance_metrics
from RenderAndCompare.geometry import (assert_bbx, assert_coord2D,
                                       assert_viewpoint)


def filter_dataset(dataset, required_object_info_fields):
    """Filter Dataset"""
    filterd_image_infos = []
    for image_info in dataset.image_infos():
        filtered_obj_infos = []
        for obj_info in image_info['object_infos']:
            if 'occlusion' in obj_info and obj_info['occlusion'] > 0.8:
                continue
            if 'truncation' in obj_info and obj_info['truncation'] > 0.8:
                continue
            # If any field is not present skip
            if any((field not in obj_info for field in required_object_info_fields)):
                continue
            filtered_obj_infos.append(obj_info)
        if filtered_obj_infos:
            image_info['object_infos'] = filtered_obj_infos
            filterd_image_infos.append(image_info)
    dataset.set_image_infos(filterd_image_infos)


def run_inference(weights_file, net, input_dataset):
    """Run inference with already initalized net with a new set of weights"""
    net.copy_from(weights_file)
    net.layers[0].generate_datum_ids()

    num_of_images = input_dataset.num_of_images()
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
    result_dataset = ImageDataset(input_dataset.name())
    result_dataset.set_rootdir(input_dataset.rootdir())
    result_dataset.set_metainfo(input_dataset.metainfo().copy())

    # Add weight file and its md5 checksum to metainfo
    result_dataset.metainfo()['weights_file'] = weights_file
    result_dataset.metainfo()['weights_file_md5'] = md5(open(weights_file, 'rb').read()).hexdigest()

    # Set the image level fields
    for input_im_info in input_dataset.image_infos():
        result_im_info = OrderedDict()
        result_im_info['image_file'] = input_im_info['image_file']
        result_im_info['image_size'] = input_im_info['image_size']
        if 'image_intrinsic' in input_im_info:
            result_im_info['image_intrinsic'] = input_im_info['image_intrinsic']
        obj_infos = []
        for input_obj_info in input_im_info['object_infos']:
            obj_info = OrderedDict()
            for field in ['id', 'category', 'score', 'bbx_visible']:
                if field in input_obj_info:
                    obj_info[field] = input_obj_info[field]
            obj_infos.append(obj_info)
        result_im_info['object_infos'] = obj_infos
        assert len(result_im_info['object_infos']) == len(input_im_info['object_infos'])
        result_dataset.add_image_info(result_im_info)

    assert result_dataset.num_of_images() == num_of_images
    assert len(net.layers[0].data_samples) == num_of_images
    for result_img_info, layer_img_info in zip(result_dataset.image_infos(), net.layers[0].data_samples):
        assert len(result_img_info['object_infos']) == len(layer_img_info['object_infos'])

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

        img_info = result_dataset.image_infos()[image_id]
        expected_num_of_rois = len(img_info['object_infos'])
        assert net.blobs['rois'].data.shape == (expected_num_of_rois, 5), "{}_{}".format(net.blobs['rois'].data.shape, expected_num_of_rois)

        for info in ["bbx_amodal", "viewpoint", "center_proj"]:
            pred_info = "pred_" + info
            if pred_info in net.blobs:
                assert net.blobs[pred_info].data.shape[0] == expected_num_of_rois

        for i, obj_info in enumerate(img_info['object_infos']):
            for info in ["bbx_amodal", "viewpoint", "center_proj"]:
                pred_info = "pred_" + info
                if pred_info in net.blobs:
                    prediction = np.squeeze(net.blobs[pred_info].data[i, ...])
                    assert_funcs[info](prediction)
                    obj_info[info] = prediction.tolist()

    return result_dataset


def prepare_dataset_for_saving(dataset):
    """This one helps with saving in a more nicer format. TODO should be part of ImageDataset"""
    for im_info in dataset.image_infos():
        for im_info_field in ['image_size', 'image_intrinsic']:
            if im_info_field in im_info:
                im_info[im_info_field] = NoIndent(im_info[im_info_field])

        for obj_info in im_info['object_infos']:
            for obj_info_field in ['bbx_visible', 'bbx_amodal', 'viewpoint', 'center_proj']:
                if obj_info_field in obj_info:
                    obj_info[obj_info_field] = NoIndent(obj_info[obj_info_field])


def try_loading_precomputed_results(result_name, weights_file):
    """Check if results already exists and return that otherwise None"""
    if osp.exists(result_name + ".json"):
        result_dataset = ImageDataset.from_json(result_name + ".json")
        # Now check if weights file and checksum matches
        md5_str = md5(open(weights_file, 'rb').read()).hexdigest()
        meta_info = result_dataset.metainfo()
        if meta_info['weights_file'] == weights_file and meta_info['weights_file_md5'] == md5_str:
            return result_dataset


def evaluate_all_weights_files(weights_files, net_file, input_dataset, gpu_id, compute_perf_metrics=True):
    """evaluate net on all weight files"""

    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # Initialize Net
    net = caffe.Net(net_file, caffe.TEST)

    # Add dataset to datalayer
    net.layers[0].add_dataset(input_dataset)

    # print data layer params
    net.layers[0].print_params()

    # Make sure we remove bad objects like tha data layer does
    filter_dataset(input_dataset, net.layers[0].required_object_info_fields)

    # Check if we have same total number of objects per image
    number_of_images = input_dataset.num_of_images()
    assert len(net.layers[0].data_samples) == number_of_images, "{} vs {}".format(len(net.layers[0].data_samples), number_of_images)
    num_of_layer_objects = sum([len(img_info['object_infos']) for img_info in net.layers[0].data_samples])
    num_of_dataset_objects = sum([len(img_info['object_infos']) for img_info in input_dataset.image_infos()])
    assert num_of_layer_objects == num_of_dataset_objects, "{} != {}".format(num_of_layer_objects, num_of_dataset_objects)

    if compute_perf_metrics:
        perf_metrics_summaries = []
        iter_regex = re.compile('iter_([0-9]*).caffemodel')

    for weights_file in weights_files:
        weight_name = osp.splitext(osp.basename(weights_file))[0]
        print 'Working with weights_file: {}'.format(weight_name)
        result_name = "{}_{}_result".format(input_dataset.name(), weight_name)

        # Check if results has been pre computed
        result_dataset = try_loading_precomputed_results(result_name, weights_file)

        if result_dataset:
            print "Skipping inference and using existing results from {}.json".format(result_name)
        else:
            # run inference
            result_dataset = run_inference(weights_file, net, input_dataset)

        if compute_perf_metrics:
            print "Evaluating results ... "
            perf_metrics_df = compute_performance_metrics(input_dataset, result_dataset)
            perf_metrics_summary_df = perf_metrics_df.describe()
            pd.set_option('display.width', 1000)
            print perf_metrics_summary_df

            perf_metrics_summary = {}
            perf_metrics_summary['iter'] = int(iter_regex.findall(weights_file)[0])
            for metric in perf_metrics_summary_df:
                perf_metrics_summary[metric + '_mean'] = perf_metrics_summary_df[metric]['mean']
                perf_metrics_summary[metric + '_std'] = perf_metrics_summary_df[metric]['std']

            perf_metrics_summaries.append(perf_metrics_summary)

        result_dataset.set_name(result_name)
        prepare_dataset_for_saving(result_dataset)
        result_dataset.write_data_to_json(result_name + ".json")
        print '--------------------------------------------------'

    # Display final summary perf metrics
    if compute_perf_metrics:
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
            arg_min = np.argmin(mean)
            plt.plot([iters[arg_min]], [mean[arg_min]], marker='o', color=color)
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
    parser.add_argument('--no_perf_metrics', dest='compute_perf_metrics', action='store_false', help="use this to disable compute_perf_metrics")
    parser.set_defaults(compute_perf_metrics=True)
    args = parser.parse_args()

    assert osp.exists(args.net_file), "Net filepath {} does not exist.".format(args.net_file)
    assert osp.exists(args.dataset_file), "ImageDataset filepath {} does not exist.".format(args.dataset_file)
    assert args.weights_files, "Weights files cannot be empty"

    for weights_file in args.weights_files:
        assert weights_file.endswith('.caffemodel'), "Weights file {} is nopt a valid Caffe weight file".format(weights_file)

    # sort the weight files
    args.weights_files.sort(key=lambda f: int(filter(str.isdigit, f)))

    print 'Loading dataset from {}'.format(args.dataset_file)
    dataset = ImageDataset.from_json(args.dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_images())

    print 'User provided {} weight files'.format(len(args.weights_files))
    evaluate_all_weights_files(args.weights_files, args.net_file, dataset, args.gpu, args.compute_perf_metrics)


if __name__ == '__main__':
    main()
