import caffe
import math
import numpy as np
import os.path as osp

from RenderAndCompare.datasets.image_loaders import BatchImageLoader

def get_predictions_on_image_files(img_files, net, weights, keys, mean, gpu_id):
    assert osp.exists(net), 'Path to test net prototxt does not exist: {}'.format(net)
    assert osp.exists(weights), 'Path to weights file does not exist: {}'.format(weights)
    assert len(mean) == 3, 'Expects mean as list of 3 numbers ([B, G, R])'

    num_of_images = len(img_files)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    net = caffe.Net(net, weights, caffe.TEST)

    data_blob_shape = net.blobs['data'].data.shape

    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'

    batch_size = data_blob_shape[0]
    im_size = [data_blob_shape[3], data_blob_shape[2]] # caffe blob are (b,c,h,w)

    image_loader = BatchImageLoader(img_files, im_size)

    predictions = {}
    for key in keys:
        blob_shape = net.blobs[key].data.shape
        assert blob_shape[0] == batch_size, 'Expects 1st channel to be batch_size'
        pred_shape = list(blob_shape)
        pred_shape[0] = num_of_images
        predictions[key] = np.zeros(tuple(pred_shape))


    mean_bgr = np.array(mean).reshape(1, 3, 1, 1)

    num_of_batches = int(math.ceil(num_of_images/float(batch_size)))
    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b+1), num_of_images)
        print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)


        for i in xrange(start_idx, end_idx):
            net.blobs['data'].data[i-start_idx, ...] = image_loader[i]

        # subtarct mean from image data blob
        net.blobs['data'].data[...] -= mean_bgr

        output = net.forward()

        for key in keys:
            predictions[key][start_idx:end_idx, :] = output[key][0:end_idx-start_idx, :]

    return predictions



