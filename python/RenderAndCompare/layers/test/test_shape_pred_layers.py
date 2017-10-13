#!/usr/bin/env python

import _init_paths
import RenderAndCompare as rac
import os.path as osp
import numpy as np
import caffe
import cv2

if __name__ == '__main__':
    default_net_file = osp.join(_init_paths.parent_dir, 'shape_pred.prototxt')

    import argparse
    description = ('Test datalayers for Crop prediction')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset", help="ImageDataset JSON file")
    parser.add_argument("-n", "--net_file", default=default_net_file, help="Net (prototxt) file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
    args = parser.parse_args()

    # init caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    net = caffe.Net(args.net_file, caffe.TEST)

    print 'Loading dataset from {}'.format(args.dataset)
    dataset = rac.datasets.ImageDataset.from_json(args.dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_images())
    num_of_images = dataset.num_of_images()

    net.layers[0].add_dataset(dataset)
    net.layers[0].generate_datum_ids()

    cv2.namedWindow('blob_image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('dataset_image', cv2.WINDOW_AUTOSIZE)

    data_blob_shape = net.blobs['data'].data.shape
    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'
    batch_size = data_blob_shape[0]
    num_of_batches = int(np.ceil(num_of_images / float(batch_size)))

    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_images)
        print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)
        output = net.forward()

        for i in xrange(start_idx, end_idx):
            annotation = dataset.image_infos()[i]

            data_blob = net.blobs['data'].data[i - start_idx]
            cv2.imshow('blob_image', net.layers[0].make_bgr8_from_blob(data_blob))

            dataset_image = cv2.imread(osp.join(dataset.rootdir(), annotation['image_file']))
            cv2.imshow('dataset_image', dataset_image)

            shape_params_blob = net.blobs['gt_shape'].data[i - start_idx]
            shape_targets_blob = net.blobs['gt_shape_target'].data[i - start_idx]

            shape_params = np.array(annotation['shape_param'], dtype=np.float32)
            assert shape_params.shape == (10,)
            assert np.isfinite(shape_params).all()

            assert np.allclose(shape_params_blob, shape_params)
            assert np.allclose(shape_targets_blob, shape_params/100.0)

            key = cv2.waitKey(args.pause)
            if key == 27:
                cv2.destroyAllWindows()
                quit = True
                break

        if quit is True:
            print 'User presessed ESC. Exiting'
            break
