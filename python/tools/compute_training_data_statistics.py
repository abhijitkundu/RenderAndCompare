#!/usr/bin/env python

import argparse

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import _init_paths
from RenderAndCompare.datasets import Dataset

def plot_viewpoint_statistics(datasets):
    """Plot viewpoint histograms"""
    print 'Computing viewpoint statistics'
    viewpoints = []
    for dataset in datasets:
        for img_info in dataset.annotations():
            for obj_info in img_info['objects']:
                if 'viewpoint' in obj_info:
                    viewpoints.append(np.array(obj_info['viewpoint'], dtype=np.float))

    if not viewpoints:
        print 'No viewpoint information found'
        return

    viewpoints = np.array(viewpoints)

    azimuths = viewpoints[:, 0]
    elevations = viewpoints[:, 1]
    tilts = viewpoints[:, 2]

    f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 7))

    f.suptitle('Viewpoint statistics', fontsize=14)

    (mu, sigma) = norm.fit(azimuths)
    _, bins, _ = ax1.hist(azimuths, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'r--', linewidth=2)
    ax1.set_title('azimuths mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax1.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(elevations)
    _, bins, _ = ax2.hist(elevations, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.plot(bins, y, 'r--', linewidth=2)
    ax2.set_title('elevations mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax2.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(tilts)
    _, bins, _ = ax3.hist(tilts, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax3.plot(bins, y, 'r--', linewidth=2)
    ax3.set_title('tilts mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax3.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

def plot_abbx_target_statistics(datasets):
    """Plot amodal bbx target histograms"""
    print 'Computing bbx_amodal_target statistics'
    abbx_targets = []
    for dataset in datasets:
        for img_info in dataset.annotations():
            for obj_info in img_info['objects']:
                if set(("bbx_visible", "bbx_amodal")) <= set(obj_info):
                    abbx = np.array(obj_info['bbx_amodal'], dtype=np.float)
                    cbbx = np.array(obj_info['bbx_visible'], dtype=np.float)
                    wh = cbbx[2:] - cbbx[:2]
                    offsets = np.array([0.0, 1.0]).reshape(2, 1)
                    abbx_target = ((abbx.reshape(2, 2) - cbbx[:2]) / wh - offsets).reshape(4,)
                    abbx_targets.append(abbx_target)

    if not abbx_targets:
        print 'No bxx_amodal_target information found'
        return

    abbx_targets = np.array(abbx_targets)

    x1_targets = abbx_targets[:, 0]
    y1_targets = abbx_targets[:, 1]
    x2_targets = abbx_targets[:, 2]
    y2_targets = abbx_targets[:, 3]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7))

    f.suptitle('Amodal bbx target statistics', fontsize=14)

    (mu, sigma) = norm.fit(x1_targets)
    _, bins, _ = ax1.hist(x1_targets, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'r--', linewidth=2)
    ax1.set_title('x1_targets mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax1.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(y1_targets)
    _, bins, _ = ax2.hist(y1_targets, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.plot(bins, y, 'r--', linewidth=2)
    ax2.set_title('y1_targets mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax2.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(x2_targets)
    _, bins, _ = ax3.hist(x2_targets, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax3.plot(bins, y, 'r--', linewidth=2)
    ax3.set_title('x2_targets mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax3.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(y2_targets)
    _, bins, _ = ax4.hist(y2_targets, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax4.plot(bins, y, 'r--', linewidth=2)
    ax4.set_title('y2_targets mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax4.axvline(0.0, color='b', linestyle='dashed', linewidth=2)


def plot_cp_target_statistics(datasets):
    """Plot center_proj bbx target histograms"""
    print 'Computing center_proj target statistics'
    cp_targets = []
    for dataset in datasets:
        for img_info in dataset.annotations():
            for obj_info in img_info['objects']:
                if set(("bbx_visible", "center_proj")) <= set(obj_info):
                    cp = np.array(obj_info['center_proj'], dtype=np.float)
                    cbbx = np.array(obj_info['bbx_visible'], dtype=np.float)
                    wh = cbbx[2:] - cbbx[:2]
                    offsets = np.array([0.5]).reshape(1, 1)
                    cp_target = ((cp.reshape(1, 2) - cbbx[:2]) / wh - offsets).reshape(2,)
                    cp_targets.append(cp_target)

    if not cp_targets:
        print 'No center_proj_target information found'
        return

    cp_targets = np.array(cp_targets)

    x_targets = cp_targets[:, 0]
    y_targets = cp_targets[:, 1]

    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(11, 7))

    f.suptitle('Center projection target statistics', fontsize=14)

    (mu, sigma) = norm.fit(x_targets)
    _, bins, _ = ax1.hist(x_targets, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'r--', linewidth=2)
    ax1.set_title('x_targets mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax1.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(y_targets)
    _, bins, _ = ax2.hist(y_targets, 60, normed=True, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.plot(bins, y, 'r--', linewidth=2)
    ax2.set_title('y_targets mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax2.axvline(0.0, color='b', linestyle='dashed', linewidth=2)


# def plot_shape_statistics(datasets):
#     print 'Computing shape statistics'
#     shape_params = []
#     for dataset in datasets:
#         for annotation in dataset.annotations():
#             if 'shape_param' in annotation:
#                 shape_params.append(np.array(annotation['shape_param'], dtype=np.float))

#     if not shape_params:
#         print 'No Shape params found'
#         return

#     shape_params = np.array(shape_params)

#     f, axes = plt.subplots(10, sharex=True, sharey=True, figsize=(7, 20))
#     f.suptitle('Shape statistics', fontsize=14)

#     for i in xrange(10):
#         values = shape_params[:, i]
#         (mu, sigma) = norm.fit(values)
#         _, bins, patches = axes[i].hist(values, 60, normed=1, facecolor='green', alpha=0.75)
#         y = mlab.normpdf(bins, mu, sigma)
#         axes[i].plot(bins, y, 'r--', linewidth=2)
#         axes[i].set_title('shape_param[%d] mean=%.3f, sigma=%.3f' % (i, mu, sigma))
#         axes[i].axvline(0.0, color='b', linestyle='dashed', linewidth=2)

def plot_center_distance_statistics(datasets):
    """Plot center_distance histograms"""
    print 'Computing center_distance statistics'
    center_distances = []
    for dataset in datasets:
        for img_info in dataset.annotations():
            for obj_info in img_info['objects']:
                if 'center_dist' in obj_info:
                    center_distances.append(obj_info['center_dist'])

    if not center_distances:
        print 'No object information found'
        return

    f = plt.figure()
    f.suptitle('center_distances statistics', fontsize=14)
    plt.hist(center_distances, bins=30, normed=False)
    plt.xlabel('center_distances', fontsize=11)

def plot_instance_count_per_image(datasets):
    """Plot num of objects per image histograms"""
    print 'Computing num_of_objects_per_image statistics'
    num_of_objects_per_image = []
    for dataset in datasets:
        for img_info in dataset.annotations():
            if 'objects' in img_info:
                num_of_objects_per_image.append(len(img_info['objects']))

    if not num_of_objects_per_image:
        print 'No object information found'
        return

    f = plt.figure()
    f.suptitle('num of objects per image statistics', fontsize=14)
    plt.hist(num_of_objects_per_image, bins=20, normed=False)
    plt.xlabel('num_of_objects_per_image', fontsize=11)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare JSON dataset files")
    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        print 'Loading dataset from {}'.format(dataset_path)
        dataset = Dataset.from_json(dataset_path)
        datasets.append(dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    # Plot image level stats
    plot_instance_count_per_image(datasets)

    # Plot object level stats
    plot_viewpoint_statistics(datasets)
    plot_abbx_target_statistics(datasets)
    plot_cp_target_statistics(datasets)
    plot_center_distance_statistics(datasets)
    # plot_shape_statistics(datasets)

    # Show all plots
    plt.show()


if __name__ == '__main__':
    main()
