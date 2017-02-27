#!/usr/bin/env python

import sys
import os.path as osp
import numpy as np

root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
import RenderAndCompare as rac

if __name__ == '__main__':
    import argparse
    description = ('Predict viewpoints from imagelist and save to a Render4CNN style view_pred file')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-n", "--net", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights", required=True, help="trained weights")
    parser.add_argument("-i", "--imagelist", default=osp.join(root_dir, 'data', 'render4cnn', 'voc12val_det_bbox','car.txt'),
                        help="textfile containing list of images")
    parser.add_argument("-o", "--output_file", default='car_pred_view.txt', help="textfile to save the results to")
    parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.939,  116.779,  123.68], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id.")

    args = parser.parse_args()


    assert osp.exists(args.imagelist), 'Path to imagelist file does not exist: {}'.format(args.imagelist)
    img_files = [x.rstrip().split(' ')[0] for x in open(args.imagelist)]
    num_of_images = len(img_files)

    print 'Predicting viewpoint for {} images'.format(num_of_images)

    predictions = rac.prediction.get_predictions_on_image_files(img_files, args.net, args.weights, ['azimuth24_prob'], args.mean_bgr, args.gpu)


    azimuth_labels = np.argmax(predictions['azimuth24_prob'], axis=1)
    azimuth_bins = predictions['azimuth24_prob'].shape[1]

    assert azimuth_bins == 24
    assert azimuth_labels.shape[0] == num_of_images

    degrees_per_bin = (360 / azimuth_bins)
    azimuths = azimuth_labels * degrees_per_bin + degrees_per_bin / 2.0

    print ('azimuth_labels = ', azimuth_labels)
    print ('azimuths = ', azimuths)

    # OUTPUT: apred epred tpred
    fout = open(args.output_file, 'w')
    for i in range(num_of_images):
        fout.write('%f %f %f\n' % (azimuths[i], 0, 0))
    fout.close()
    print 'Results written to {}'.format(args.output_file)
