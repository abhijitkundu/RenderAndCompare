#!/usr/bin/env python

import _init_paths
import RenderAndCompare as rac


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--dataset_a", required=True, type=str, help="Path to RenderAndCompare JSON dataset file")
    parser.add_argument("-b", "--dataset_b", required=True, type=str, help="Path to RenderAndCompare JSON dataset file")

    args = parser.parse_args()

    print 'Loading dataset A from {}'.format(args.dataset_a)
    datasetA = rac.datasets.ImageDataset.from_json(args.dataset_a)
    print 'Loaded {} dataset A with {} annotations'.format(datasetA.name(), datasetA.num_of_images())

    print 'Loading dataset B from {}'.format(args.dataset_b)
    datasetB = rac.datasets.ImageDataset.from_json(args.dataset_b)
    print 'Loaded {} dataset B with {} annotations'.format(datasetB.name(), datasetB.num_of_images())

    all_equal = (datasetA.data == datasetB.data)

    print 'Datsets equal: {}'.format(all_equal)
