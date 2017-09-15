#!/usr/bin/env python

import _init_paths
import RenderAndCompare as rac
import os.path as osp
import numpy as np
import time

if __name__ == '__main__':
    import argparse
    description = ('Predict shape and pose for articulated objects')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights_file", required=True, help="trained weights")
    parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU Id.")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-o", "--output", default='results.json', help="output json filepath")

    args = parser.parse_args()

    print 'Loading dataset from {}'.format(args.dataset)
    dataset = rac.datasets.Dataset.from_json(args.dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    image_files = []
    cropping_boxes = []

    shape_params_gt = np.zeros((dataset.num_of_annotations(), dataset.metainfo()['shape_param_dimension']))
    pose_params_gt = np.zeros((dataset.num_of_annotations(), dataset.metainfo()['pose_param_dimension']))

    for i in xrange(dataset.num_of_annotations()):
        annotation = dataset.annotations()[i]
        img_path = osp.join(dataset.rootdir(), annotation['image_file'])

        visible_bbx = np.array(annotation['bbx_visible'], dtype=np.float)  # gt box (only visible path)
        # TODO Check for bbx validity

        cropping_boxes.append(visible_bbx)
        image_files.append(img_path)

        shape_params_gt[i, ...] = annotation['shape_param']
        pose_params_gt[i, ...] = annotation['pose_param']

    num_of_images = len(image_files)
    print 'Predicting viewpoint for {} images'.format(num_of_images)
    t0 = time.time()
    predictions = rac.prediction.get_predictions_on_image_croppings(image_files,
                                                                    cropping_boxes,
                                                                    args.net_file,
                                                                    args.weights_file,
                                                                    ['pred_shape_target', 'pred_pose_target'],
                                                                    args.mean_bgr,
                                                                    args.gpu)

    t1 = time.time()
    total = t1 - t0
    print 'Took {} seconds'.format(total)

    assert shape_params_gt.shape == predictions['pred_shape_target'].shape
    assert pose_params_gt.shape == predictions['pred_pose_target'].shape

    mean_shape_L1 = np.mean(np.sum(np.absolute(shape_params_gt - predictions['pred_shape_target']), axis=1))
    mean_pose_L1 = np.mean(np.sum(np.absolute(pose_params_gt - predictions['pred_pose_target']), axis=1))

    print 'mean_shape_L1 = {}'.format(mean_shape_L1)
    print 'mean_pose_L1 = {}'.format(mean_pose_L1)

    for i in xrange(dataset.num_of_annotations()):
        dataset.annotations()[i]['shape_param'] = predictions['pred_shape_target'][i, ...].tolist()
        dataset.annotations()[i]['pose_param'] = predictions['pred_pose_target'][i, ...].tolist()

    print 'Saving results at {}'.format(args.output)
    dataset.write_data_to_json(args.output)
