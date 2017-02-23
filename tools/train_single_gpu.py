#!/usr/bin/env python
"""
Trains a model using one GPU.
"""
import _init_paths
import caffe
import os.path as osp
import warnings


def train(solver_proto, initialization, gpu_id):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    solver = caffe.get_solver(solver_proto)

    if initialization is not None:
        assert osp.exists(
            initialization), 'Path to weights/solverstate does not exist: {}'.format(initialization)
        if initialization.endswith('.solverstate'):
            print 'Restoring solverstate from {}'.format(initialization)
            solver.restore(initialization)
        elif initialization.endswith('.caffemodel'):
            print 'Initializing weights from {}'.format(initialization)
            solver.net.copy_from(initialization)
        else:
            raise ValueError(
                'ERROR: {} is not supported for initailization'.format(initialization))
    else:
        warnings.warn(
            "Warning: No initialization provided. Training from scratch.")

    # train according to solver params
    solver.solve()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--solver", required=True,
                        help="Solver solver_proto definition.")
    parser.add_argument(
        "--init", help="Initialization weights or Solver state to restore from")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu Id.")
    args = parser.parse_args()

    # print args.init

    train(args.solver, args.init, args.gpu)
