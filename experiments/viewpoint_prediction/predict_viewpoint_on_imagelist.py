#!/usr/bin/env python

import sys
import os.path as osp
import numpy as np
import time

root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
import RenderAndCompare as rac

if __name__ == '__main__':
    import argparse
    description = ('Predict viewpoints from imagelist and save to a Render4CNN style view_pred file')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights_file", required=True, help="trained weights")
    parser.add_argument("-i", "--imagelist", default=osp.join(root_dir, 'data', 'render4cnn', 'voc12val_det_bbox', 'car_faster_rcnn.txt'),
                        help="textfile containing list of images")
    parser.add_argument("-o", "--output_file", default='car_pred_view.txt', help="textfile to save the results to")
    parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id.")

    args = parser.parse_args()

    assert osp.exists(args.imagelist), 'Path to imagelist file does not exist: {}'.format(args.imagelist)
    img_files = [x.rstrip().split(' ')[0] for x in open(args.imagelist)]
    num_of_images = len(img_files)

    print 'Predicting viewpoint for {} images'.format(num_of_images)

    t0 = time.time()
    predictions = rac.prediction.get_predictions_on_image_files(img_files,
                                                                args.net_file,
                                                                args.weights_file,
                                                                ['azimuth_pred', 'elevation_pred', 'tilt_pred'],
                                                                args.mean_bgr,
                                                                args.gpu)
    t1 = time.time()
    total = t1 - t0
    print 'Took {} seconds'.format(total)

    if 'azimuth_pred' in predictions:
        azimuths = np.squeeze(predictions['azimuth_pred'])
    else:
        azimuths = np.zeros(num_of_images)

    if 'elevation_pred' in predictions:
        elevations = np.squeeze(predictions['elevation_pred'])
    else:
        elevations = np.zeros(num_of_images)

    if 'tilt_pred' in predictions:
        tilts = np.squeeze(predictions['tilt_pred'])
    else:
        tilts = np.zeros(num_of_images)

    # OUTPUT: apred epred tpred
    fout = open(args.output_file, 'w')
    for i in range(num_of_images):
        fout.write('%f %f %f\n' % (azimuths[i], elevations[i], tilts[i]))
    fout.close()
    print 'Results written to {}'.format(args.output_file)
