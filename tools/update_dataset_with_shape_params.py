#!/usr/bin/env python

import _init_paths
import RenderAndCompare as rac
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dataset", required=True, type=str, help="Path to Input RenderAndCompare JSON dataset file")
    parser.add_argument("-o", "--output_dataset", default='dataset_with_shape.json', type=str, help="Output path for updated RenderAndCompare JSON dataset file")
    parser.add_argument("-s", "--shape_file", required=True, type=str, help="Shape data file in HDF5 format")

    args = parser.parse_args()

    print 'Loading shape data from {}'.format(args.shape_file)
    with h5py.File(args.shape_file, 'r') as hf:
        model_names = list(hf['model_names'][:])
        encoded_training_data = hf['encoded_training_data'][:]
        num_of_shapes = len(model_names)
        assert num_of_shapes == encoded_training_data.shape[1]
    print 'Found {} shapes with dimension {}'.format(num_of_shapes, encoded_training_data.shape[0])

    shape_params = {}
    for i in xrange(num_of_shapes):
        shape_params[model_names[i]] = encoded_training_data[:, i].tolist()
    assert len(shape_params) == num_of_shapes

    print 'Loading dataset from {}'.format(args.input_dataset)
    dataset = rac.datasets.Dataset.from_json(args.input_dataset)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    print 'Updating shape data ...'
    annotations_with_shape = []
    for annotation in tqdm(dataset.annotations()):
        shape_id = annotation['image_file'].partition('/')[0]
        if shape_id in shape_params:
            annotation['shape_param'] = shape_params[shape_id]
            annotations_with_shape.append(annotation)

    dataset.set_annotations(annotations_with_shape)
    print 'Finished updating annotations. Found {} matches ({} images/shape)'.format(dataset.num_of_annotations(), dataset.num_of_annotations() / num_of_shapes)

    print 'Saving updated dataset to {}'.format(args.output_dataset)
    dataset.write_data_to_json(args.output_dataset)
