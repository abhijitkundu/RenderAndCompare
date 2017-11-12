#!/usr/bin/env python
import math
import os.path as osp
import sys

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
    num_of_images = dataset.num_of_images()

    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    net = caffe.Net(args.net_file, args.weights_file, caffe.TEST)

    net.layers[0].add_dataset(dataset)
    net.layers[0].generate_datum_ids()

    data_blob_shape = net.blobs['data'].data.shape

    assert len(data_blob_shape) == 4, 'Expects 4D data blob'
    assert data_blob_shape[1] == 3, 'Expects 2nd channel to be 3 for BGR image'

    batch_size = data_blob_shape[0]
    num_of_batches = int(math.ceil(num_of_images / float(batch_size)))
    for b in xrange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_images)
        print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)
        output = net.forward()

        # for i in xrange(start_idx, end_idx):
        #     print viewpoints_gt[i][0], output['azimuth_pred_mode'][i - start_idx], output['azimuth_pred'][i - start_idx], output['azimuth_cs'][i - start_idx]
        # # print output['azimuth_cs']
        # # print output['azimuth_pred']
