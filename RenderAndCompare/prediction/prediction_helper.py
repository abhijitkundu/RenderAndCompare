import caffe
import math
import numpy as np
import os.path as osp
import cv2

from RenderAndCompare.datasets import BatchImageLoader


def get_predictions_on_image_files(img_files, net_file, weights_file, requested_keys, mean, gpu_id):
    assert osp.exists(net_file), 'Path to test net prototxt does not exist: {}'.format(net_file)
    assert osp.exists(weights_file), 'Path to weights file does not exist: {}'.format(weights_file)
    assert len(mean) == 3, 'Expects mean as list of 3 numbers ([B, G, R])'

    num_of_images = len(img_files)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    net = caffe.Net(net_file, weights_file, caffe.TEST)

    data_blob_shape = net.blobs['data'].data.shape

    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'

    batch_size = data_blob_shape[0]
    im_size = [data_blob_shape[3], data_blob_shape[2]]  # caffe blob are (b,c,h,w)

    image_loader = BatchImageLoader(im_size, img_files)

    predictions = {}
    for key in requested_keys:
        if key in net.blobs:
            blob_shape = net.blobs[key].data.shape
            assert blob_shape[0] == batch_size, 'Expects 1st channel to be batch_size'
            pred_shape = list(blob_shape)
            pred_shape[0] = num_of_images
            predictions[key] = np.zeros(tuple(pred_shape))
        else:
            print 'Requested key {} not available in network'.format(key)

    mean_bgr = np.array(mean).reshape(1, 3, 1, 1)

    num_of_batches = int(math.ceil(num_of_images / float(batch_size)))
    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_images)
        print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)

        for i in xrange(start_idx, end_idx):
            net.blobs['data'].data[i - start_idx, ...] = image_loader[i]

        # subtarct mean from image data blob
        net.blobs['data'].data[...] -= mean_bgr

        output = net.forward()

        for key in predictions:
            predictions[key][start_idx:end_idx, :] = output[key][0:end_idx - start_idx, :]

    return predictions


def get_predictions_on_image_boxes(image, boxes, net, requested_keys, mean):
    num_of_boxes = len(boxes)
    assert num_of_boxes != 0

    data_blob_shape = net.blobs['data'].data.shape

    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'

    batch_size = data_blob_shape[0]
    im_size = [data_blob_shape[3], data_blob_shape[2]]  # caffe blob are (b,c,h,w)

    predictions = {}
    for key in requested_keys:
        if key in net.blobs:
            blob_shape = net.blobs[key].data.shape
            assert blob_shape[0] == batch_size, 'Expects 1st channel to be batch_size'
            pred_shape = list(blob_shape)
            pred_shape[0] = num_of_boxes
            predictions[key] = np.zeros(tuple(pred_shape))
        else:
            print 'Requested key {} not available in network'.format(key)

    image = image - np.asarray(mean)  # subtract mean bgr from each pixel

    W = image.shape[1]
    H = image.shape[0]

    num_of_batches = int(math.ceil(num_of_boxes / float(batch_size)))
    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_boxes)

        for i in xrange(start_idx, end_idx):
            bbx = boxes[i]
            bbx_min = np.maximum.reduce([np.array([0, 0]), np.floor(bbx[:2]).astype(int)])
            bbx_max = np.minimum.reduce([np.array([W - 1, H - 1]), np.floor(bbx[2:]).astype(int)])

            assert np.all((bbx_max - bbx_min) > 0), 'Bad bbx: {} -- {}'.format(bbx_min, bbx_max)

            bbx_image = image[bbx_min[1]:bbx_max[1] + 1, bbx_min[0]:bbx_max[0] + 1, :]
            assert bbx_image.ndim == 3

            bbx_image = cv2.resize(bbx_image, (im_size[0], im_size[1]), interpolation=cv2.INTER_LINEAR)
            bbx_image = bbx_image.transpose((2, 0, 1))  # move image channels to outermost dimension

            net.blobs['data'].data[i - start_idx, ...] = bbx_image

        output = net.forward()

        for key in predictions:
            predictions[key][start_idx:end_idx, :] = output[key][0:end_idx - start_idx, :]

    return predictions
