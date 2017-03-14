#!/usr/bin/env python

import _init_paths
import os.path as osp
import RenderAndCompare as rac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


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

    bbx_tfms = []
    for dataset in datasets:
        for annotation in dataset.annotations():
            bbx_a = np.array(annotation['bbx_amodal'], dtype=np.float)
            bbx_c = np.array(annotation['bbx_crop'], dtype=np.float)
            aTc = [(bbx_c[0] - bbx_a[0]) / bbx_a[2], (bbx_c[1] - bbx_a[1]) / bbx_a[3], bbx_c[2] / bbx_a[2], bbx_c[3] / bbx_a[3]]
            bbx_tfms.append(aTc)

    bbx_tfms = np.array(bbx_tfms)
    print bbx_tfms.shape

    x_offsets = bbx_tfms[:, 0]
    y_offsets = bbx_tfms[:, 1]
    w_offsets = np.log(bbx_tfms[:, 2])
    h_offsets = np.log(bbx_tfms[:, 3])

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    (mu, sigma) = norm.fit(x_offsets)
    n, bins, patches = ax1.hist(x_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'r--', linewidth=2)
    ax1.set_title('x offset mean=%.3f, sigma=%.3f' % (mu, sigma))

    (mu, sigma) = norm.fit(y_offsets)
    n, bins, patches = ax2.hist(y_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax2.plot(bins, y, 'r--', linewidth=2)
    ax2.set_title('y offset mean=%.3f, sigma=%.3f' % (mu, sigma))

    (mu, sigma) = norm.fit(w_offsets)
    n, bins, patches = ax3.hist(w_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax3.plot(bins, y, 'r--', linewidth=2)
    ax3.set_title('w offset mean=%.3f, sigma=%.3f' % (mu, sigma))

    (mu, sigma) = norm.fit(h_offsets)
    n, bins, patches = ax4.hist(h_offsets, 60, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    ax4.plot(bins, y, 'r--', linewidth=2)
    ax4.set_title('h offset mean=%.3f, sigma=%.3f' % (mu, sigma))

    plt.show()
