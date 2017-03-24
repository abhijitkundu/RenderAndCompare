#!/usr/bin/env python

import _init_paths
import RenderAndCompare as rac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


def plot_bbx_statistics(datasets):
    print 'Computing bbx statistics'
    bbx_tfms = []
    for dataset in datasets:
        for annotation in dataset.annotations():
            if set(("bbx_amodal", "bbx_crop")) <= set(annotation):
                bbx_a = np.array(annotation['bbx_amodal'], dtype=np.float)
                bbx_c = np.array(annotation['bbx_crop'], dtype=np.float)
                aTc = [(bbx_c[0] - bbx_a[0]) / bbx_a[2], (bbx_c[1] - bbx_a[1]) / bbx_a[3], bbx_c[2] / bbx_a[2], bbx_c[3] / bbx_a[3]]
                bbx_tfms.append(aTc)

    if not bbx_tfms:
        print 'No bbx information found'
        return

    bbx_tfms = np.array(bbx_tfms)

    x_offsets = bbx_tfms[:, 0]
    y_offsets = bbx_tfms[:, 1]
    w_offsets = np.log(bbx_tfms[:, 2])
    h_offsets = np.log(bbx_tfms[:, 3])

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7))

    f.suptitle('Bbx statistics', fontsize=14)

    (mu, sigma) = norm.fit(x_offsets)
    n, bins, patches = ax1.hist(x_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'r--', linewidth=2)
    ax1.set_title('x offset mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax1.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(y_offsets)
    n, bins, patches = ax2.hist(y_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.plot(bins, y, 'r--', linewidth=2)
    ax2.set_title('y offset mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax2.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(w_offsets)
    n, bins, patches = ax3.hist(w_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax3.plot(bins, y, 'r--', linewidth=2)
    ax3.set_title('w offset mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax3.axvline(0.0, color='b', linestyle='dashed', linewidth=2)

    (mu, sigma) = norm.fit(h_offsets)
    n, bins, patches = ax4.hist(h_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax4.plot(bins, y, 'r--', linewidth=2)
    ax4.set_title('h offset mean=%.3f, sigma=%.3f' % (mu, sigma))
    ax1.axvline(0.0, color='b', linestyle='dashed', linewidth=2)


def plot_shape_statistics(datasets):
    print 'Computing shape statistics'
    shape_params = []
    for dataset in datasets:
        for annotation in dataset.annotations():
            if 'shape_param' in annotation:
                shape_params.append(np.array(annotation['shape_param'], dtype=np.float))

    if not shape_params:
        print 'No Shape params found'
        return

    shape_params = np.array(shape_params)

    f, axes = plt.subplots(10, sharex=True, sharey=True, figsize=(7, 20))
    f.suptitle('Shape statistics', fontsize=14)

    for i in xrange(10):
        values = shape_params[:, i]
        (mu, sigma) = norm.fit(values)
        n, bins, patches = axes[i].hist(values, 60, normed=1, facecolor='green', alpha=0.75)
        y = mlab.normpdf(bins, mu, sigma)
        axes[i].plot(bins, y, 'r--', linewidth=2)
        axes[i].set_title('shape_param[%d] mean=%.3f, sigma=%.3f' % (i, mu, sigma))
        axes[i].axvline(0.0, color='b', linestyle='dashed', linewidth=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare JSON dataset files")
    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        print 'Loading dataset from {}'.format(dataset_path)
        dataset = rac.datasets.Dataset.from_json(dataset_path)
        datasets.append(dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    plot_bbx_statistics(datasets)
    plot_shape_statistics(datasets)

    # Show all plots
    plt.show()
