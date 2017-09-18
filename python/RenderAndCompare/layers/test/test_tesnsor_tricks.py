#!/usr/bin/env python

import _init_paths
import caffe
import numpy as np
import os.path as osp
from pprint import pprint
caffe.set_mode_cpu()

caffe.set_random_seed(13397)
np.random.seed(0)

net = caffe.Net('TensorStuff.prototxt', caffe.TRAIN)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

N = net.blobs['input_data'].num
assert net.blobs['gt_viewpoint'].data.shape == (N, 3)

net.blobs['input_data'].data[...] = np.random.rand(*net.blobs['input_data'].data.shape)
net.blobs['gt_viewpoint'].data[...] = np.random.uniform(-np.pi, np.pi, (N, 3))
net.params['scale_fc_viewpoint_3x2'][0].data[...] = np.array([1., 2, 3])

out = net.forward()

assert np.allclose(net.blobs['fc_viewpoint_6'].data.reshape(4, 3, 2), net.blobs['fc_viewpoint_3x2'].data)

assert np.allclose(net.blobs['fc_viewpoint_3x2'].data * np.array([1., 2, 3]).reshape(1, 3, 1), net.blobs['fc_viewpoint_3x2_by_T'].data)
assert np.allclose(np.sum(net.blobs['prob_viewpoint_3x2_by_T'].data, axis=2), np.ones((4, 3)))

# print out['loss_viewpoint_label']
pprint(out)

tatal_loss = float(out['loss_azimuth_label'] + out['loss_elevation_label'] + out['loss_tilt_label'])
print tatal_loss
print out['loss_viewpoint_label'] * 3
