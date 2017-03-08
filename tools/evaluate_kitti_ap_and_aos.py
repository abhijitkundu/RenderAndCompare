#!/usr/bin/env python

import _init_paths
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call


if __name__ == '__main__':

    kitti_object_gt_dir = osp.join(_init_paths.root_dir, 'data', 'kitti', 'KITTI-Object', 'training', 'label_2')
    assert osp.exists(kitti_object_gt_dir), 'KITTI Object dir "{}" soes not exist'.format(kitti_object_gt_dir)

    splits_file_default = osp.join(_init_paths.root_dir, 'data', 'kitti', 'splits', '3dvp_val.txt')
    evalauation_app = osp.join(_init_paths.root_dir, 'data', 'kitti', 'cpp_evaluate', 'evaluate_kitti_object')

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--result_dir", required=True, help="Path to Results folder")
    parser.add_argument("-g", "--gt_dir", default=kitti_object_gt_dir, help="KITTI GT label directory")
    parser.add_argument("-s", "--split_file", default=splits_file_default, help="Path to split file")

    args = parser.parse_args()

    for arg in vars(args):
        print "\t{} \t= {}".format(arg, getattr(args, arg))
        print "------------------------------------------------------------"

    call([evalauation_app, args.gt_dir, args.result_dir, args.split_file])

    detection_results_file = osp.join(args.result_dir, 'plot', 'car_detection.txt')
    orientation_results_file = osp.join(args.result_dir, 'plot', 'car_orientation.txt')

    if osp.exists(detection_results_file):
        detection_results = np.loadtxt(detection_results_file)
        easy_ap = 100.0 * np.mean(detection_results[0:41:4, 1])
        moderate_ap = 100.0 * np.mean(detection_results[0:41:4, 2])
        hard_ap = 100.0 * np.mean(detection_results[0:41:4, 3])
        print 'AP_easy={}, AP_modeerate={}, AP_hard={}'.format(easy_ap, moderate_ap, hard_ap)

        plt.figure(1)
        plt.plot(detection_results[:, 0], detection_results[:, 1], color="red", linewidth=2.0, label="Easy AP = %%%.02f" % easy_ap)
        plt.plot(detection_results[:, 0], detection_results[:, 2], color="green", linewidth=2.0, label="Moderate AP = %%%.02f" % moderate_ap)
        plt.plot(detection_results[:, 0], detection_results[:, 3], color="blue", alpha=0.6, label="Hard AP = %%%.02f" % hard_ap)
        plt.xlabel('recall')
        plt.xlabel('precision')
        plt.legend(loc='lower left')
        plt.title('Detection Performance (AP)')
        plt.savefig(osp.join(args.result_dir, 'ap_plot.png'), bbox_inches='tight')
        plt.show()

    if osp.exists(orientation_results_file):
        orientation_results = np.loadtxt(orientation_results_file)
        easy_aos = 100.0 * np.mean(orientation_results[0:41:4, 1])
        moderate_aos = 100.0 * np.mean(orientation_results[0:41:4, 2])
        hard_aos = 100.0 * np.mean(orientation_results[0:41:4, 3])
        print 'AOS_easy={}, AOS_moderate={}, AOS_hard={}'.format(easy_aos, moderate_aos, hard_aos)

        plt.figure(2)
        plt.plot(orientation_results[:, 0], orientation_results[:, 1], color="red", linewidth=2.0, label="Easy AOS = %%%.02f" % easy_aos)
        plt.plot(orientation_results[:, 0], orientation_results[:, 2], color="green", linewidth=2.0, label="Moderate AOS = %%%.02f" % moderate_aos)
        plt.plot(orientation_results[:, 0], detection_results[:, 3], color="blue", linewidth=2.0, label="Hard AOS = %%%.02f" % hard_aos)
        plt.xlabel('recall')
        plt.xlabel('precision')
        plt.legend(loc='lower left')
        plt.title('Orientation + Detection Performance (AOS)')
        plt.savefig(osp.join(args.result_dir, 'aos_plot.png'), bbox_inches='tight')
        plt.show()
