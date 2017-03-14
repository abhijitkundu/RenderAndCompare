#!/usr/bin/env python
"""
Times a net model
"""
import _init_paths
import RenderAndCompare as rac
import caffe
import os.path as osp


def time(net, iters):
    fprop = []
    bprop = []
    total = caffe.Timer()
    for _ in range(len(net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())

    def show_time():
        s = '\n'
        for i in range(len(net.layers)):
            s += 'forw %3d %8s ' % (i, net._layer_names[i])
            s += ': %.2f\n' % fprop[i].ms
        for i in range(len(net.layers) - 1, -1, -1):
            s += 'back %3d %8s ' % (i, net._layer_names[i])
            s += ': %.2f\n' % bprop[i].ms
        s += 'solver total: %.2f\n' % total.ms
        caffe.log(s)

    net.before_forward(lambda layer: fprop[layer].start())
    net.after_forward(lambda layer: fprop[layer].stop())
    net.before_backward(lambda layer: bprop[layer].start())
    net.after_backward(lambda layer: bprop[layer].stop())
    total.start()
    for i in xrange(iters):
        net.forward()
        net.backward()
    total.stop()
    show_time()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("net_file", help="network model proto definition.")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    parser.add_argument("-i", "--iters", type=int, default=10, help="Number of test iterations")
    parser.add_argument("-d", "--dataset", help="Dataset JSON file")
    args = parser.parse_args()

    caffe.init_log()
    caffe.log('Using GPU# %s' % str(args.gpu))

    # init caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    net = caffe.Net(args.net_file, caffe.TRAIN)

    if args.dataset is not None:
        print 'Loading dataset from {}'.format(args.dataset)
        dataset = rac.datasets.Dataset.from_json(args.dataset)
        print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())
        net.layers[0].add_dataset(dataset)
        net.layers[0].generate_datum_ids()

    print 'Will now run Fwd and Bkwd for {} times'.format(args.iters)
    time(net, args.iters)
