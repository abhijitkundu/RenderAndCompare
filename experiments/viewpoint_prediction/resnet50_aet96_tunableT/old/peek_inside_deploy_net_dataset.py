#!/usr/bin/env python
import math
import os.path as osp
import sys

import numpy as np

import caffe
import RenderAndCompare as rac

root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, root_dir)

if __name__ == '__main__':
    import argparse
    description = ('Predict viewpoints from imagelist')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights_file", required=True, help="trained weights")
    parser.add_argument("-d", "--dataset", default=osp.join(root_dir, 'data', 'pascal3D', 'pascal3d_voc2012_val_easy',
                                                            'pascal3d_voc2012_val_easy_car.json'), help="ImageDataset JSON file")
    parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084],
                        type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id.")
    # parser.add_argument("keys", nargs='+', help="Keys to query")

    args = parser.parse_args()

    print 'Loading dataset from {}'.format(args.dataset)
    dataset = rac.datasets.ImageDataset.from_json(args.dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_images())

    image_files = []
    viewpoints_gt = []
    for i in xrange(dataset.num_of_images()):
        annotation = dataset.image_infos()[i]
        img_path = osp.join(dataset.rootdir(), annotation['image_file'])
        image_files.append(img_path)

        viewpoint = annotation['viewpoint'][:3]
        viewpoints_gt.append(viewpoint)

    num_of_images = len(image_files)

    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    net = caffe.Net(args.net_file, args.weights_file, caffe.TEST)

    data_blob_shape = net.blobs['data'].data.shape

    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'

    batch_size = data_blob_shape[0]
    im_size = [data_blob_shape[3], data_blob_shape[2]]  # caffe blob are (b,c,h,w)

    image_loader = rac.datasets.BatchImageLoader(im_size, image_files)
    mean_bgr = np.array(args.mean_bgr).reshape(1, 3, 1, 1)

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
        for i in xrange(start_idx, end_idx):
            print viewpoints_gt[i][0], output['azimuth_pred_mode'][i - start_idx], output['azimuth_pred_expectation'][i - start_idx], output['azimuth_cs'][i - start_idx]
        # print output['azimuth_cs']
        # print output['azimuth_pred']
