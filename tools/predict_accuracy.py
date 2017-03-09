#!/usr/bin/env python
"""
Tests a model for accuracy
"""

import _init_paths
import RenderAndCompare as rac
import caffe
import os.path as osp


def eval_net(net_file, weights, dataset, test_iters=77):
    assert osp.exists(net_file), 'Path to test net prototxt does not exist: {}'.format(net_file)
    assert osp.exists(weights), 'Path to weights file does not exist: {}'.format(weights)

    test_net = caffe.Net(net_file, weights, caffe.TEST)

    for dataset in datasets:
        test_net.layers[0].add_dataset(dataset)
    test_net.layers[0].generate_datum_ids()

    print 'Evaluating accuracy with {} test iterations'.format(test_iters)

    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['accuracy']
    accuracy /= test_iters
    return accuracy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare JSON dataset files")
    parser.add_argument("-n", "--net_file", required=True, help="Solver solver_proto definition.")
    parser.add_argument("-w", "--weights_file", required=True, help="Initialization weights or Solver state to restore from")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-i", "--iters", type=int, default=10, help="Number of test iterations")

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

    accuracy = eval_net(args.net_file, args.weights_file, dataset, args.iters)
    print 'Test set Accuracy: %3.1f%%' % (100 * accuracy, )
