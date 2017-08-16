#!/usr/bin/env python
"""
Tests a model for accuracy
"""

import _init_paths
import RenderAndCompare as rac
import numpy as np
import os.path as osp
import caffe
import math
import tqdm


def eval_net(net_file, weights, datasets):
    assert osp.exists(net_file), 'Path to net prototxt does not exist: {}'.format(net_file)
    assert osp.exists(weights), 'Path to weights file does not exist: {}'.format(weights)

    net = caffe.Net(net_file, weights, caffe.TEST)

    for dataset in datasets:
        net.layers[0].add_dataset(dataset)
    num_of_data_points = net.layers[0].generate_datum_ids()

    data_blob_shape = net.blobs['data'].data.shape
    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'

    batch_size = data_blob_shape[0]
    num_of_batches = int(math.ceil(num_of_data_points / float(batch_size)))

    print 'Evaluating for {} batches'.format(num_of_batches)

    accuracy_outputs = {}
    for b in tqdm.trange(num_of_batches):
        output = net.forward()
        # store all accuracy outputs
        for key in [key for key in output if "accuracy" in key]:
            if key in accuracy_outputs:
                accuracy_outputs[key].append(output[key])
            else:
                accuracy_outputs[key] = [output[key]]

    for key in accuracy_outputs:
        accuracy = np.mean(accuracy_outputs[key])
        print 'Test set {}: {:.2f}'.format(key, (100. * accuracy))
    return accuracy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare JSON dataset files")
    parser.add_argument("-n", "--net_file", required=True, help="network prototxt file.")
    parser.add_argument("-w", "--weights_file", required=True, help="Initialization weights or Solver state to restore from")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")

    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        print 'Loading dataset from {}'.format(dataset_path)
        dataset = rac.datasets.Dataset.from_json(dataset_path)
        datasets.append(dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    # init caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    eval_net(args.net_file, args.weights_file, datasets)
    # print 'Test set Accuracy: %3.1f%%' % (100 * accuracy, )
