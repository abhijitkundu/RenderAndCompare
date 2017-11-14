#!/usr/bin/env python
"""
Trains a model using one or more GPUs.
"""
import os.path as osp
import warnings
from multiprocessing import Process

import _init_paths
import caffe
from RenderAndCompare.datasets import ImageDataset


def train(
        solver,  # solver proto definition
        initialization,  # weights or solver snapshot to restore from
        datasets,
        gpus  # list of device ids
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    # caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver, initialization, datasets, gpus, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def solve(proto, initialization, datasets, gpus, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])        
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)

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
    solver.net.layers[0].print_params()
    solver.net.layers[0].generate_datum_ids()

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    solver.step(solver.param.max_iter)

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("datasets", nargs='+', help="Path to RenderAndCompare JSON dataset files")
    parser.add_argument("-s", "--solver", required=True, help="Solver proto definition.")
    parser.add_argument("-i", "--init", help="Initialization weights or Solver state to restore from")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
    args = parser.parse_args()

    datasets = []
    for dataset_path in args.datasets:
        print 'Loading dataset from {}'.format(dataset_path)
        dataset = ImageDataset.from_json(dataset_path)
        datasets.append(dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_images())

    train(args.solver, args.init, datasets, args.gpus)


if __name__ == '__main__':
    main()
