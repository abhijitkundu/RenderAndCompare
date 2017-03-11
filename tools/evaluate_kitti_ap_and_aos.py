#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
import os.path as osp
import numpy as np
import math
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
    parser.add_argument("-d", "--display", type=int, default=1, help="Set to zero if you do not wanna display plots")

    args = parser.parse_args()

    print "------------------------------------------------------------"
    for arg in vars(args):
        print "\t{} \t= {}".format(arg, getattr(args, arg))
    print "------------------------------------------------------------"

    call([evalauation_app, args.gt_dir, args.result_dir, args.split_file])

    detection_results_file = osp.join(args.result_dir, 'plot', 'car_detection.txt')
    orientation_results_file = osp.join(args.result_dir, 'plot', 'car_orientation.txt')

    results_name = osp.basename(args.result_dir)

    assert osp.exists(detection_results_file), 'Requires at-least the detection results'

    detection_results = np.loadtxt(detection_results_file)
    easy_ap = 100.0 * np.mean(detection_results[0:41:4, 1])
    moderate_ap = 100.0 * np.mean(detection_results[0:41:4, 2])
    hard_ap = 100.0 * np.mean(detection_results[0:41:4, 3])

    easy_ap_tag = 'AP={0:.2f}%'.format(easy_ap)
    mod_ap_tag = 'AP={0:.2f}%'.format(moderate_ap)
    hard_ap_tag = 'AP={0:.2f}%'.format(hard_ap)

    easy_tag = "Easy: {}".format(easy_ap_tag)
    mod_tag = "Moderate: {}".format(mod_ap_tag)
    hard_tag = "Hard: {}".format(hard_ap_tag)

    print easy_tag, mod_tag, hard_tag

    plt.figure(1)
    plt.plot(detection_results[:, 0], detection_results[:, 1], color="red", linewidth=2.0, label=easy_tag)
    plt.plot(detection_results[:, 0], detection_results[:, 2], color="green", linewidth=2.0, label=mod_tag)
    plt.plot(detection_results[:, 0], detection_results[:, 3], color="blue", alpha=0.6, label=hard_tag)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='lower left')
    plt.title('Detection Performance (AP) with ' + results_name)
    plt.savefig(osp.join(args.result_dir, 'plot', 'ap_plot.png'), bbox_inches='tight')

    if osp.exists(orientation_results_file):
        orientation_results = np.loadtxt(orientation_results_file)
        easy_aos = 100.0 * np.mean(orientation_results[0:41:4, 1])
        moderate_aos = 100.0 * np.mean(orientation_results[0:41:4, 2])
        hard_aos = 100.0 * np.mean(orientation_results[0:41:4, 3])

        easy_aae = math.degrees(math.acos(2 * (easy_aos / easy_ap) - 1))
        mod_aae = math.degrees(math.acos(2 * (moderate_aos / moderate_ap) - 1))
        hard_aae = math.degrees(math.acos(2 * (hard_aos / hard_ap) - 1))

        easy_aos_tag = 'AOS={0:.2f}%'.format(easy_aos)
        mod_aos_tag = 'AOS={0:.2f}%'.format(moderate_aos)
        hard_aos_tag = 'AOS={0:.2f}%'.format(hard_aos)

        easy_aae_tag = u'AAE={0:.4f}\u00b0'.format(easy_aae)
        mod_aae_tag = u'AAE={0:.4f}\u00b0'.format(mod_aae)
        hard_aae_tag = u'AAE={0:.4f}\u00b0'.format(hard_aae)

        easy_tag = u"Easy: {} {}".format(easy_aos_tag, easy_aae_tag)
        mod_tag = u"Moderate: {} {}".format(mod_aos_tag, mod_aae_tag)
        hard_tag = u"Hard: {} {}".format(hard_aos_tag, hard_aae_tag)

        print easy_tag, mod_tag, hard_tag

        plt.figure(2)
        plt.plot(orientation_results[:, 0], orientation_results[:, 1], color="red", linewidth=2.0, label=easy_tag)
        plt.plot(orientation_results[:, 0], orientation_results[:, 2], color="green", linewidth=2.0, label=mod_tag)
        plt.plot(orientation_results[:, 0], detection_results[:, 3], color="blue", linewidth=2.0, label=hard_tag)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend(loc='lower left')
        plt.title('Orientation Performance (AOS) with ' + results_name)
        plt.savefig(osp.join(args.result_dir, 'plot', 'aos_plot.png'), bbox_inches='tight')

    if args.display:
        plt.show()
