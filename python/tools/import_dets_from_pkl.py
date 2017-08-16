#!/usr/bin/env python
import _init_paths
import RenderAndCompare as rac
import os.path as osp
from os import makedirs
import cPickle
from scipy import io as sio


def write_dets_as_mat(boxes, filename):
    """
    car_dets_prunned = [x[x[:, 4] > score_thresh, :] for x in car_dets]
    write_dets_as_mat(car_dets_prunned, 'car_dets.mat')
    """
    sio.savemat(filename, {'boxes': boxes})


if __name__ == '__main__':
    import argparse
    description = ('Import detections from pickle file')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("pickle_file", help="Path to Pickle file")
    parser.add_argument("-s", "--score_threshold", default=0.0, type=float, help="score threshold")
    parser.add_argument("-o", "--output_folder", type=str, help="output folder")

    args = parser.parse_args()

    assert osp.exists(args.pickle_file)

    with open(args.pickle_file, 'r') as f:
        all_dets = cPickle.load(f)

    car_dets = all_dets[1]
    num_of_images = len(car_dets)

    if args.output_folder is not None:
        print 'Will save results in {}'.format(args.output_folder)
        if not osp.exists(args.output_folder):
            makedirs(args.output_folder)

    num_of_dets = 0
    for image_id in xrange(num_of_images):
        car_detections_current_image = list(car_dets[image_id])
        car_dets_current_image_prunned = [x for x in car_detections_current_image if (x[4] > args.score_threshold)]
        num_of_dets += len(car_dets_current_image_prunned)

        if args.output_folder is not None:
            objects = []
            for det in car_dets_current_image_prunned:
                obj = {}
                obj['type'] = 'Car'
                obj['bbox'] = list(det[:4])
                obj['score'] = det[4]
                objects.append(obj)

            out_label_filepath = osp.join(args.output_folder, '{:06d}.txt'.format(image_id))
            rac.datasets.write_kitti_object_labels(objects, out_label_filepath)

    print 'Number of detections = {:,}'.format(num_of_dets)
