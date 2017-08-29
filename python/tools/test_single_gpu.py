#!/usr/bin/env python
"""
Trains a model using one GPU.
"""

import os.path as osp

import numpy as np
import tqdm

import _init_paths
import caffe
from RenderAndCompare.datasets import Dataset, NoIndent


def test(dataset, net_file, weights_file, gpu_id):
    """Run inference"""
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    net = caffe.Net(net_file, weights_file, caffe.TEST)

    net.layers[0].add_dataset(dataset)
    net.layers[0].generate_datum_ids()

    data_samples = net.layers[0].data_samples
    num_of_data_samples = len(data_samples)
    batch_size = net.layers[0].batch_size
    num_of_batches = int(np.ceil(num_of_data_samples / float(batch_size)))

    assert len(net.layers[0].image_loader) == dataset.num_of_annotations()

    # clear the objects
    for image_info in dataset.annotations():
        image_info['objects'][:] = []
        image_info['image_size'] = NoIndent(image_info['image_size'])
        image_info['image_intrinsic'] = NoIndent(image_info['image_intrinsic'])

    accuracy_outputs = {}

    print 'Evaluating for {} batches with {} imaes per batch.'.format(num_of_batches, batch_size)
    for b in tqdm.trange(num_of_batches):
        start_idx = batch_size * b
        end_idx = min(batch_size * (b + 1), num_of_data_samples)
        # print 'Working on batch: %d/%d (Image# %d - %d)' % (b, num_of_batches, start_idx, end_idx)
        output = net.forward()

        # store all accuracy outputs
        for key in [key for key in output if "accuracy" in key]:
            if key in accuracy_outputs:
                accuracy_outputs[key].append(output[key])
            else:
                accuracy_outputs[key] = [output[key]]

        for i in xrange(start_idx, end_idx):
            image_id = data_samples[i]['image_id']
            image_info = dataset.annotations()[image_id]

            object_info = {}

            viewpoint_pred = np.squeeze(output['viewpoint_pred'][i - start_idx, ...])
            assert (viewpoint_pred >= -np.pi).all() and (viewpoint_pred < np.pi).all()
            object_info['viewpoint'] = NoIndent(viewpoint_pred.tolist())
            object_info['bbx_visible'] = NoIndent(data_samples[i]['bbx_visible'].tolist())

            image_info['objects'].append(object_info)

    for key in accuracy_outputs:
        accuracy = np.mean(accuracy_outputs[key])
        print 'Test set {}: {:.2f}'.format(key, (100. * accuracy))

    num_of_objects = sum([len(image_info['objects']) for image_info in dataset.annotations()])
    assert num_of_objects == num_of_data_samples
    return dataset


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Test on dataset")

    parser.add_argument("dataset_file", nargs=1, help="Path to RenderAndCompare JSON dataset file")
    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-w", "--weights_file", required=True, help="trained weights")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    args = parser.parse_args()

    assert osp.exists(args.net_file), "Net filepath {} does not exist.".format(args.net_file)
    assert osp.exists(args.weights_file), "weights filepath {} does not exist.".format(args.weights_file)

    print 'Loading dataset from {}'.format(args.dataset_file[0])
    dataset = Dataset.from_json(args.dataset_file[0])
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    result_dataset = test(dataset, args.net_file, args.weights_file, args.gpu)
    out_json_filename = result_dataset.name() + '_result.json'
    print 'Saving results to {}'.format(out_json_filename)
    dataset.write_data_to_json(out_json_filename)


if __name__ == '__main__':
    main()
