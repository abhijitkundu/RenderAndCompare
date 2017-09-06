#!/usr/bin/env python
"""
Trains a model using one GPU.
"""

import os.path as osp
from collections import OrderedDict

import numpy as np
import tqdm

import _init_paths
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from RenderAndCompare.datasets import Dataset, NoIndent

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

    # Set the image level fields
    for input_im_info in input_dataset.annotations():
        result_im_info = OrderedDict()
        result_im_info['image_file'] = input_im_info['image_size']
        result_im_info['image_size'] = NoIndent(input_im_info['image_size'])
        result_im_info['image_intrinsic'] = NoIndent(input_im_info['image_intrinsic'])
        result_im_info['objects'] = []
        result_dataset.add_annotation(result_im_info)

    assert result_dataset.num_of_annotations() == input_dataset.num_of_annotations()

    accuracy_outputs = {}

    print 'Evaluating for {} batches with {} imaes per batch.'.format(num_of_batches, batch_size)
    for b in tqdm.trange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_data_samples)
        # print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)
        output = net.forward()

        # store all accuracy outputs
        for key in [key for key in output if "accuracy" in key]:
            if key in accuracy_outputs:
                accuracy_outputs[key].append(output[key])
            else:
                accuracy_outputs[key] = [output[key]]

        for i in xrange(start_idx, end_idx):
            image_id = data_samples[i]['image_id']
            image_info = result_dataset.annotations()[image_id]

            object_info = {}

            viewpoint_pred = np.squeeze(output['viewpoint_pred'][i - start_idx, ...])
            assert (viewpoint_pred >= -np.pi).all() and (viewpoint_pred < np.pi).all()
            object_info['viewpoint'] = NoIndent(viewpoint_pred.tolist())
            object_info['bbx_visible'] = NoIndent(data_samples[i]['bbx_visible'].tolist())

            image_info['objects'].append(object_info)

    for key in accuracy_outputs:
        accuracy = np.mean(accuracy_outputs[key])
        print 'Test set {}: {:.2f}'.format(key, (100. * accuracy))

    result_num_of_objects = sum([len(image_info['objects']) for image_info in result_dataset.annotations()])
    assert result_num_of_objects == num_of_data_samples
    return result_dataset


def test_all_weights_files(weights_files, net_file, input_dataset, gpu_id):
    """Run inference on all weight files"""
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # Initialize Net
    net = caffe.Net(net_file, caffe.TEST)

    # Add dataset to datalayer
    net.layers[0].add_dataset(input_dataset)

    for weights_file in weights_files:
        weight_name = osp.splitext(osp.basename(weights_file))[0]
        print 'Working with weights_file: {}'.format(weight_name)
        result_dataset = test_single_weights_file(weights_file, net, input_dataset)
        out_json_filename = "{}_{}_result.json".format(result_dataset.name(), weight_name)
        result_dataset.write_data_to_json(out_json_filename)
        print '--------------------------------------------------'


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
