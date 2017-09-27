#!/usr/bin/env python

import os.path as osp
from time import time
import caffe

if __name__ == '__main__':
    import argparse
    description = ('Test datalayer')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("net_file", help="Net (prototxt) file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations")

    args = parser.parse_args()

    # init caffe
    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    else:
        caffe.set_mode_cpu()

    assert osp.exists(args.net_file), 'Net file "{}" do not exist'.format(args.net_file)
    print 'Loading net model from {}'.format(args.net_file)

    net = caffe.Net(args.net_file, caffe.TEST)

    # for each layer, show the output shape
    print "{0:13} | Blob shape".format("layer_names")
    print "----------------------------------"
    for layer_name, blob in net.blobs.iteritems():
        print "{0:15} {1}".format(layer_name, blob.data.shape)

    print "\n"

    # for each layer, show the params shape
    print "{0:13} | params shape".format("layer_names")
    print "----------------------------------"
    for layer_name, param in net.params.iteritems():
        param_shapes = [str(p.data.shape) for p in param]
        print "{0:15} {1}".format(layer_name, '; '.join(param_shapes))

    print 'Running forward pass for {} iterations'.format(args.iterations)
    start = time()
    for i in xrange(args.iterations):
        output = net.forward()
    end = time()
    print "Took {}s ".format(end - start)
