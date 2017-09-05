#!/usr/bin/env python

import os.path as osp

import cv2
import numpy as np

import _init_paths
import caffe
import RenderAndCompare as rac

if __name__ == '__main__':
    import argparse
    description = ('Test datalayer')
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
    dataset = rac.datasets.Dataset.from_json(args.dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    net.layers[0].add_dataset(dataset)
    net.layers[0].generate_datum_ids()

    data_samples = net.layers[0].data_samples

    cv2.namedWindow('blob_image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('full_image', cv2.WINDOW_AUTOSIZE)

    image_blob_shape = net.blobs['input_image'].data.shape
    assert len(image_blob_shape) == 4, 'Expects 4D data blob'
    assert image_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'
    batch_size = image_blob_shape[0]
    num_of_batches = int(np.ceil(len(data_samples) / float(batch_size)))

    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), len(data_samples))
        print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)
        output = net.forward()

        for i in xrange(start_idx, end_idx):
            data_sample = data_samples[i]

            image_blob = net.blobs['input_image'].data[i - start_idx]
            cv2.imshow('blob_image', net.layers[0].make_bgr8_from_blob(image_blob))

            bbx_amodal_blob = net.blobs['gt_bbx_amodal'].data[i - start_idx]
            bbx_crop_blob = net.blobs['gt_bbx_crop'].data[i - start_idx]
            # aTc_blob = net.blobs['crop_target'].data[i - start_idx, ...]

            bbx_a = data_sample['bbx_amodal'].astype(np.float32)
            bbx_v = data_sample['bbx_visible'].astype(np.float32)

            assert np.allclose(bbx_amodal_blob, bbx_a)
            assert np.allclose(bbx_crop_blob, bbx_v)

            full_image = net.layers[0].image_loader[data_sample['image_id']].copy()
            cv2.rectangle(full_image, (bbx_a[0], bbx_a[1]), (bbx_a[2], bbx_a[3]), (0, 255, 0), 1)
            cv2.rectangle(full_image, (bbx_crop_blob[0], bbx_crop_blob[1]), (bbx_crop_blob[2], bbx_crop_blob[3]), (0, 0, 255), 1)
            cv2.imshow('full_image', full_image)

            viewpoint = data_sample['viewpoint']
            viewpoint_blob = net.blobs['gt_viewpoint'].data[i - start_idx]
            assert np.allclose(viewpoint, viewpoint_blob)

            viewpoint_label = net.blobs['gt_viewpoint_label'].data[i - start_idx]
            assert (viewpoint_label >= 0).all() and (viewpoint_label < 96).all()
            assert viewpoint_label[0] == net.blobs['gt_vp_azimuth_label'].data[i - start_idx]
            assert viewpoint_label[1] == net.blobs['gt_vp_elevation_label'].data[i - start_idx]
            assert viewpoint_label[2] == net.blobs['gt_vp_tilt_label'].data[i - start_idx]

            # # Only for testing perfect transformation
            # pred_bbx_amodal = net.blobs['pred_bbx_amodal'].data[i - start_idx, ...]
            # assert np.allclose(pred_bbx_amodal, bbx_a, rtol=1e-04, atol=1e-06), 'pred_bbx_amodal={} and bbx_a={} are different'.format(pred_bbx_amodal, bbx_a)
            # iou = net.blobs['ious'].data[i - start_idx, ...]
            # assert np.allclose(iou, 1.0)

            key = cv2.waitKey(args.pause)
            if key == 27:
                cv2.destroyAllWindows()
                quit = True
                break

        if quit is True:
            print 'User presessed ESC. Exiting'
            break
