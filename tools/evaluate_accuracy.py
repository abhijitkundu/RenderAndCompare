#!/usr/bin/env python
"""
Tests a model for accuracy
"""

import _init_paths
import caffe
import os.path as osp

def eval_net(net, weights, test_iters=77):
    assert osp.exists(net), 'Path to test net prototxt does not exist: {}'.format(net)
    assert osp.exists(weights), 'Path to weights file does not exist: {}'.format(weights)

    test_net = caffe.Net(net, weights, caffe.TEST)

    print 'Evaluating accuracy with {} test iterations'.format(test_iters)
    
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['accuracy']
    accuracy /= test_iters
    return accuracy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--net", required=True, help="Solver solver_proto definition.")
    parser.add_argument("--weights", required=True, help="Initialization weights")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("--iters", type=int, default=0, help="Number of test iterations")
    args = parser.parse_args()



    # init caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    

    accuracy = eval_net(args.net, args.weights, args.iters)
    print 'Test set Accuracy: %3.1f%%' % (100*accuracy, )