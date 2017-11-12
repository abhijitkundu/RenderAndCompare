#!/usr/bin/env python

import os.path as osp

import cv2
import numpy as np

import _init_paths
import caffe
import RenderAndCompare as rac

if __name__ == '__main__':
    default_net_file = osp.join(_init_paths.parent_dir, 'crop_pred_offset_based.prototxt')

    import argparse
    description = ('Test datalayers for Crop prediction')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset", help="ImageDataset JSON file")
    parser.add_argument("-n", "--net_file", default=default_net_file, help="Net (prototxt) file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
    args = parser.parse_args()

    # init caffe
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

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

            bbx_amodal_blob = net.blobs['gt_bbx_amodal'].data[i - start_idx]
            bbx_crop_blob = net.blobs['gt_bbx_crop'].data[i - start_idx]

            bbx_a = np.array(annotation['bbx_amodal'], dtype=np.float32)
            bbx_c = np.array(annotation['bbx_crop'], dtype=np.float32)

            assert np.allclose(bbx_amodal_blob, bbx_a)
            assert np.allclose(bbx_crop_blob, bbx_c)

            if 'crop_target' in net.blobs:
                aTc = [(bbx_c[0] - bbx_a[0]) / bbx_a[2], (bbx_c[1] - bbx_a[1]) / bbx_a[3], np.log(bbx_c[2] / bbx_a[2]), np.log(bbx_c[3] / bbx_a[3])]
                aTc_blob = net.blobs['crop_target'].data[i - start_idx, ...]
                assert np.allclose(aTc_blob, aTc)
            elif all(k in net.blobs for k in ("position_offset", "size_offset")):
                aTc = [(bbx_c[0] - bbx_a[0]) / bbx_a[2], (bbx_c[1] - bbx_a[1]) / bbx_a[3], bbx_c[2] / bbx_a[2], bbx_c[3] / bbx_a[3]]
                aTc_blob = np.hstack((net.blobs['position_offset'].data[i - start_idx, ...], net.blobs['size_offset'].data[i - start_idx, ...]))
                assert np.allclose(aTc_blob, aTc), 'aTc_blob={} aTc={}'.format(aTc_blob, aTc)
            else:
                print 'No bbx transformation targets found'

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
