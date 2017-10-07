#!/usr/bin/env python
"""
Evaluate a RCNN style model
"""

import os.path as osp
import re
from collections import OrderedDict
from hashlib import md5

import numpy as np
import pandas as pd
import tqdm

import _init_paths
import caffe
from RenderAndCompare.datasets import Dataset, NoIndent
from RenderAndCompare.geometry import assert_viewpoint, assert_bbx, assert_coord2D


def test_single_weights_file(weights_file, net, input_dataset):
    """Test already initalized net with a new set of weights"""
    net.copy_from(weights_file)
    net.layers[0].generate_datum_ids()

    input_num_of_objects = sum([len(image_info['objects']) for image_info in input_dataset.annotations()])
    assert net.layers[0].curr_data_ids_idx == 0
    assert net.layers[0].number_of_datapoints() == input_num_of_objects
    assert net.layers[0].data_ids == range(input_num_of_objects)

    data_samples = net.layers[0].data_samples
    num_of_data_samples = len(data_samples)
    batch_size = net.layers[0].batch_size
    num_of_batches = int(np.ceil(num_of_data_samples / float(batch_size)))

    assert len(net.layers[0].image_loader) == input_dataset.num_of_annotations()

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
        result_im_info['image_size'] = NoIndent(input_im_info['image_size'])
        result_im_info['image_intrinsic'] = NoIndent(input_im_info['image_intrinsic'])
        result_im_info['objects'] = []
        result_dataset.add_annotation(result_im_info)

    assert result_dataset.num_of_annotations() == input_dataset.num_of_annotations()

    assert_funcs = {
        "viewpoint": assert_viewpoint,
        "bbx_visible": assert_bbx,
        "bbx_amodal": assert_bbx,
        "center_proj": assert_coord2D,
    }

    performance_metric = {}

    print 'Evaluating for {} batches with {} imaes per batch.'.format(num_of_batches, batch_size)
    for b in tqdm.trange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_data_samples)
        # print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)
        output = net.forward()

        # store all accuracy outputs
        for key in [key for key in output if any(x in key for x in ["accuracy", "iou", "error"])]:
            assert np.squeeze(output[key]).shape == (), "Expects {} output to be scalar but got {}".format(key, output[key].shape)
            current_batch_accuracy = float(np.squeeze(output[key]))
            if key in performance_metric:
                performance_metric[key].append(current_batch_accuracy)
            else:
                performance_metric[key] = [current_batch_accuracy]

        for i in xrange(start_idx, end_idx):
            image_id = data_samples[i]['image_id']
            image_info = result_dataset.annotations()[image_id]

            object_info = OrderedDict()

            # since we are not changing cetegory orid it is directly copied
            object_info['id'] = data_samples[i]['id']
            object_info['category'] = data_samples[i]['category']

            # since we are not predicting bbx_visible, it is directly copied
            object_info['bbx_visible'] = NoIndent(data_samples[i]['bbx_visible'].tolist())

            for info in ["bbx_amodal", "viewpoint", "center_proj"]:
                pred_info = "pred_" + info
                if pred_info in net.blobs:
                    prediction = np.squeeze(net.blobs[pred_info].data[i - start_idx, ...])
                    assert_funcs[info](prediction)
                    object_info[info] = NoIndent(prediction.tolist())

            image_info['objects'].append(object_info)

    for key in sorted(performance_metric):
        performance_metric[key] = np.mean(performance_metric[key])
        print 'Test set {}: {:.4f}'.format(key, performance_metric[key])

    regex = re.compile('iter_([0-9]*).caffemodel')
    performance_metric['iter'] = int(regex.findall(weights_file)[0])

    result_num_of_objects = sum([len(image_info['objects']) for image_info in result_dataset.annotations()])
    assert result_num_of_objects == num_of_data_samples
    return result_dataset, performance_metric


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

    # store accuracies for each weight file here
    performance_metrics = []

    for weights_file in weights_files:
        weight_name = osp.splitext(osp.basename(weights_file))[0]
        print 'Working with weights_file: {}'.format(weight_name)
        result_dataset, performance_metric = test_single_weights_file(weights_file, net, input_dataset)
        performance_metrics.append(performance_metric)
        result_name = "{}_{}_result".format(result_dataset.name(), weight_name)
        result_dataset.set_name(result_name)
        result_dataset.write_data_to_json(result_name + ".json")
        print '--------------------------------------------------'

    perf_metrics_df = pd.DataFrame(performance_metrics).set_index('iter')
    print 'Saving performance metrics to {}.csv'.format(input_dataset.name() + '_all_metrics')
    perf_metrics_df.to_csv(input_dataset.name() + '_all_metrics' + '.csv')

    if len(weights_files) > 1:
        accuracy_df = perf_metrics_df.filter(regex='accuracy')
        error_df = perf_metrics_df.filter(regex='error')
        if not accuracy_df.empty:
            print 'Saving accuracy plot to {}.png'.format(input_dataset.name() + '_accuracy')
            accuracy_df.plot().get_figure().savefig(input_dataset.name() + '_accuracy' + '.png')
        if not error_df.empty:
            print 'Saving error plot to {}.png'.format(input_dataset.name() + '_error')
            error_df.plot().get_figure().savefig(input_dataset.name() + '_error' + '.png')


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
