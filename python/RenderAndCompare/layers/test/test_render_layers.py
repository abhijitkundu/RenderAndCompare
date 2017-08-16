#!/usr/bin/env python

import _init_paths
import RenderAndCompare as rac
import os.path as osp
import numpy as np
import caffe
import cv2

if __name__ == '__main__':
    default_net_file = osp.join(_init_paths.parent_dir, 'model_render.prototxt')
    import argparse
    description = ('Test datalayer')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-n", "--net_file", default=default_net_file, help="Net (prototxt) file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
    args = parser.parse_args()

    # init caffe
    # caffe.set_mode_gpu()
    # caffe.set_device(args.gpu)
    caffe.set_mode_cpu()

    assert osp.exists(args.net_file), 'Net file "{}" do not exist'.format(args.net_file)
    net = caffe.Net(args.net_file, caffe.TEST)

    blob_shape = net.blobs['rendered_output'].data.shape
    print blob_shape
    # assert len(blob_shape) == 4, 'Expects 4D data blob'