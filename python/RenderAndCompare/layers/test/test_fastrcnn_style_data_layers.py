#!/usr/bin/env python

import os.path as osp

import cv2
import numpy as np

import caffe
import _init_paths
from RenderAndCompare.datasets import Dataset
from RenderAndCompare.geometry import assert_viewpoint, assert_bbx, assert_coord2D

if __name__ == '__main__':
    import argparse
    description = ('Test Fast-RCNN style datalayer')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset", help="Dataset JSON file")
    parser.add_argument("-n", "--net_file", required=True, help="Net (prototxt) file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
    args = parser.parse_args()

    # init caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    assert osp.exists(args.net_file), 'Net file "{}" do not exist'.format(args.net_file)
    net = caffe.Net(args.net_file, caffe.TEST)

    print 'Loading dataset from {}'.format(args.dataset)
    dataset = Dataset.from_json(args.dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    net.layers[0].add_dataset(dataset)
    net.layers[0].generate_datum_ids()

    assert net.layers[0].number_of_datapoints() == dataset.num_of_annotations()
    number_of_images = dataset.num_of_annotations()

    cv2.namedWindow('blob_image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('original_image', cv2.WINDOW_AUTOSIZE)

    image_blob_shape = net.blobs['input_image'].data.shape
    assert len(image_blob_shape) == 4, 'Expects 4D data blob'
    assert image_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'
    batch_size = image_blob_shape[0]
    num_of_batches = int(np.ceil(dataset.num_of_annotations() / float(batch_size)))

    exit_loop = False
    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), number_of_images)
        print 'Working on batch: %d/%d (Images# %d - %d)' % (b, num_of_batches, start_idx, end_idx)

        # Run forward pass
        output = net.forward()

        # Get image_scales and image_flippings
        image_scales = net.blobs['image_scales'].data
        image_flippings = net.blobs['image_flippings'].data.astype(np.bool)
        assert image_scales.shape == image_flippings.shape == (batch_size,)

        # Get roi_blob and from that determine number_of_rois
        roi_blob = net.blobs['roi'].data
        assert roi_blob.ndim == 2 and roi_blob.shape[1] == 5

        number_of_rois = roi_blob.shape[0]
        for roi_id in xrange(number_of_rois):
            roi_batch_index = roi_blob[roi_id, 0]
            assert 0 <= roi_batch_index <= batch_size
            assert_bbx(roi_blob[roi_id, -4:])

        # Check the bbx blobs
        bbx_amodal_blob = net.blobs['gt_bbx_amodal'].data
        bbx_crop_blob = net.blobs['gt_bbx_crop'].data
        for bbx_blob in [bbx_amodal_blob, bbx_crop_blob]:
            assert bbx_blob.shape == (number_of_rois, 4)
            for roi_id in xrange(number_of_rois):
                assert_bbx(bbx_blob[roi_id, :])

        # Check the center proj blobs
        center_proj_blob = net.blobs['gt_center_proj'].data
        assert center_proj_blob.shape == (number_of_rois, 2)

        # Check vp blobs
        vp_blob = net.blobs['gt_viewpoint'].data
        assert vp_blob.shape == (number_of_rois, 3), "Weird vp shape = {}".format(vp_blob)
        assert (vp_blob >= -np.pi).all() and (vp_blob < np.pi).all()

        for i in xrange(start_idx, end_idx):
            original_image = cv2.imread(osp.join(dataset.rootdir(), dataset.annotations()[i]['image_file']))
            cv2.imshow('original_image', original_image)

            image_blob = net.blobs['input_image'].data[i - start_idx]
            image_blob_bgr8 = net.layers[0].make_bgr8_from_blob(image_blob).copy()

            for roi_id in xrange(roi_blob.shape[0]):
                roi_batch_index = roi_blob[roi_id, 0]
                if roi_batch_index == (i - start_idx):
                    bbx_roi = roi_blob[roi_id, -4:].astype(np.float32)
                    cv2.rectangle(image_blob_bgr8, tuple(bbx_roi[:2]), tuple(bbx_roi[2:]), (0, 255, 0), 1)

            cv2.imshow('blob_image', image_blob_bgr8)
            cv2.displayOverlay('blob_image', 'Flipped' if image_flippings[i - start_idx] else 'Original')

            key = cv2.waitKey(args.pause)
            if key == 27:
                cv2.destroyAllWindows()
                exit_loop = True
                break

        if exit_loop is True:
            print 'User presessed ESC. Exiting'
            break