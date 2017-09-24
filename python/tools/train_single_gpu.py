#!/usr/bin/env python
"""
Trains a model using one GPU.
"""
import os.path as osp
import warnings

import _init_paths
import caffe
from RenderAndCompare.datasets import Dataset


def train(solver_proto, datasets, initialization, gpu_id):
    """Train a network"""
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    solver = caffe.get_solver(solver_proto)

    if initialization is not None:
        assert osp.exists(initialization), 'Path to weights/solverstate does not exist: {}'.format(initialization)
        if initialization.endswith('.solverstate'):
            print 'Restoring solverstate from {}'.format(initialization)
            solver.restore(initialization)
        elif initialization.endswith('.caffemodel'):
            print 'Initializing weights from {}'.format(initialization)
            solver.net.copy_from(initialization)
        else:
            raise ValueError('ERROR: {} is not supported for initailization'.format(initialization))
    else:
        warnings.warn("Warning: No initialization provided. Training from scratch.")

    for dataset in datasets:
        solver.net.layers[0].add_dataset(dataset)
    solver.net.layers[0].generate_datum_ids()

    # train according to solver params
    solver.solve()


def main():
    """Mian function"""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare JSON dataset files")
    parser.add_argument("-s", "--solver", required=True, help="Solver solver_proto definition.")
    parser.add_argument("-i", "--init", help="Initialization weights or Solver state to restore from")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        print 'Loading dataset from {}'.format(dataset_path)
        dataset = Dataset.from_json(dataset_path)
        datasets.append(dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    train(args.solver, datasets, args.init, args.gpu)


if __name__ == '__main__':
    main()
