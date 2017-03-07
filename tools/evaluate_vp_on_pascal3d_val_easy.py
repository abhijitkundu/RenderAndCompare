#!/usr/bin/env python
"""
Evaluates viewpoint accuracy with grounttruth boxes on pascal3d val easy
"""

import _init_paths
import RenderAndCompare as rac
import numpy as np
import os.path as osp
import time
import math

if __name__ == '__main__':
    import argparse
    description = ('Predict viewpoints from imagelist and save to a Render4CNN style view_pred file')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-n", "--net", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights", required=True, help="trained weights")
    parser.add_argument("-d", "--dataset", default=osp.join(_init_paths.root_dir, 'data', 'pascal3D', 'pascal3d_voc2012_val_easy', 'pascal3d_voc2012_val_easy_car.json'), help="Dataset JSON file")
    parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id.")
    parser.add_argument("-o", "--output_file", help="textfile to save the results to")

    args = parser.parse_args()

    print 'Loading dataset from {}'.format(args.dataset)
    dataset = rac.datasets.Dataset.from_json(args.dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    image_files = []
    viewpoints_gt = []
    for i in xrange(dataset.num_of_annotations()):
        annotation = dataset.annotations()[i]
        img_path = osp.join(dataset.rootdir(), annotation['image_file'])
        image_files.append(img_path)

        viewpoint = annotation['viewpoint'][:3]
        viewpoints_gt.append(viewpoint)

    num_of_images = len(image_files)

    print 'Predicting viewpoint for {} images'.format(num_of_images)

    t0 = time.time()
    predictions = rac.prediction.get_predictions_on_image_files(image_files,
                                                                args.net,
                                                                args.weights,
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

    viewpoints_pred = zip(azimuths, elevations, tilts)

    geodesic_errors = rac.geometry.compute_geodesic_errors(viewpoints_gt, viewpoints_pred)

    acc_pi_by_6 = float((geodesic_errors < (np.pi / 6)).sum()) / num_of_images
    acc_pi_by_12 = float((geodesic_errors < (np.pi / 12)).sum()) / num_of_images
    acc_pi_by_24 = float((geodesic_errors < (np.pi / 24)).sum()) / num_of_images

    med_err_deg = math.degrees(np.median(geodesic_errors))

    print 'med_err_deg= {}, acc_pi_by_6= {}, acc_pi_by_12= {}, acc_pi_by_24= {}'.format(med_err_deg, acc_pi_by_6, acc_pi_by_12, acc_pi_by_24)

    if args.output_file is not None:
        fout = open(args.output_file, 'w')
        for i in range(num_of_images):
            # OUTPUT: apred epred tpred
            fout.write('%f %f %f\n' % (azimuths[i], elevations[i], tilts[i]))
        fout.close()
        print 'Results written to {}'.format(args.output_file)
