#!/usr/bin/env python

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import _init_paths
import RenderAndCompare as rac


def plot_param_statistics(datasets, param_key='body_shape'):
    print 'Computing statistics for param: {}'.format(param_key)
    params = []
    for dataset in datasets:
        for annotation in dataset.image_infos():
            if param_key in annotation:
                params.append(np.array(annotation[param_key], dtype=np.float))

    if not params:
        print 'No {} params found'.format(param_key)
        return

    params = np.array(params)
    print 'params.shape() = {}'.format(params.shape)
    print 'params.min = {} params.max = {}'.format(params.min(), params.max())

    f, axes = plt.subplots(10, sharex=True, sharey=True, figsize=(7, 20))
    f.suptitle('{} statistics'.format(param_key), fontsize=14)

    for i in xrange(10):
        values = params[:, i]
        (mu, sigma) = norm.fit(values)
        n, bins, patches = axes[i].hist(values, 60, normed=1, facecolor='green', alpha=0.75)
        y = mlab.normpdf(bins, mu, sigma)
        axes[i].plot(bins, y, 'r--', linewidth=2)
        axes[i].set_title('%s[%d] mean=%.3f, sigma=%.3f' % (param_key, i, mu, sigma))
        axes[i].axvline(0.0, color='b', linestyle='dashed', linewidth=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Articulated Object Stats')
    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare Articulated object JSON dataset files")
    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        print 'Loading dataset from {}'.format(dataset_path)
        dataset = rac.datasets.ImageDataset.from_json(dataset_path)
        datasets.append(dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_images())

    plot_param_statistics(datasets, 'shape_param')
    plot_param_statistics(datasets, 'pose_param')

    # Show all plots
    plt.show()
