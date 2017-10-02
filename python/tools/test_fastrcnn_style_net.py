#!/usr/bin/env python
"""
Tests a FastRCNN style model
"""

import os.path as osp
from collections import OrderedDict

import numpy as np
import tqdm

import _init_paths
import caffe
from RenderAndCompare.datasets import Dataset, NoIndent
from RenderAndCompare.geometry import assert_viewpoint, assert_bbx, assert_coord2D


def test_single_weights_file(weights_file, net, input_dataset):
    """Test already initalized net with a new set of weights"""
    net.copy_from(weights_file)
    net.layers[0].generate_datum_ids()

    num_of_images = input_dataset.num_of_annotations()
    assert net.layers[0].curr_data_ids_idx == 0
    assert net.layers[0].number_of_datapoints() == num_of_images
    assert net.layers[0].data_ids == range(num_of_images)

    assert len(net.layers[0].image_loader) == num_of_images
    assert len(net.layers[0].data_samples) == num_of_images
    assert net.layers[0].rois_per_image < 0, "rois_per_image need to be dynamic for testing"
    assert net.layers[0].imgs_per_batch == 1, "We only support one image per batch while testing"

    # Create Result dataset
    result_dataset = Dataset(input_dataset.name())
    result_dataset.set_rootdir(input_dataset.rootdir())
    result_dataset.set_metainfo(input_dataset.metainfo().copy())

    # Set the image level fields
    for input_im_info in input_dataset.annotations():
        result_im_info = OrderedDict()
        result_im_info['image_file'] = input_im_info['image_file']
        result_im_info['image_size'] = NoIndent(input_im_info['image_size'])
        result_im_info['image_intrinsic'] = NoIndent(input_im_info['image_intrinsic'])
        obj_infos = []
        for input_obj_info in input_im_info['objects']:
            obj_info = OrderedDict()
            obj_info['id'] = input_obj_info['id']
            obj_info['category'] = input_obj_info['category']
            obj_info['bbx_visible'] = NoIndent(input_obj_info['bbx_visible'])
            obj_infos.append(obj_info)
        result_im_info['objects'] = obj_infos
        assert len(result_im_info['objects']) == len(input_im_info['objects'])
        result_dataset.add_annotation(result_im_info)

    assert result_dataset.num_of_annotations() == num_of_images

    assert_funcs = {
        "viewpoint": assert_viewpoint,
        "bbx_visible": assert_bbx,
        "bbx_amodal": assert_bbx,
        "center_proj": assert_coord2D,
    }

    print 'Running inference for {} images.'.format(num_of_images)
    for image_id in tqdm.trange(num_of_images):
        # Run forward pass
        output = net.forward()

        img_info = result_dataset.annotations()[image_id]
        expected_num_of_rois = len(img_info['objects'])
        assert net.blobs['rois'].data.shape == (expected_num_of_rois, 5)

        for info in ["bbx_amodal", "viewpoint", "center_proj"]:
            pred_info = "pred_" + info
            if pred_info in net.blobs:
                assert net.blobs[pred_info].data.shape[0] == expected_num_of_rois

        for i, obj_info in enumerate(img_info['objects']):
            for info in ["bbx_amodal", "viewpoint", "center_proj"]:
                pred_info = "pred_" + info
                if pred_info in net.blobs:
                    prediction = np.squeeze(net.blobs[pred_info].data[i, ...])
                    assert_funcs[info](prediction)
                    obj_info[info] = NoIndent(prediction.tolist())

    return result_dataset


def test_all_weights_files(weights_files, net_file, input_dataset, gpu_id):
    """Run inference on all weight files"""
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # Initialize Net
    net = caffe.Net(net_file, caffe.TEST)

    # Add dataset to datalayer
    net.layers[0].add_dataset(input_dataset)

    for weights_file in weights_files:
        weight_name = osp.splitext(osp.basename(weights_file))[0]
        print 'Working with weights_file: {}'.format(weight_name)
        result_dataset = test_single_weights_file(weights_file, net, input_dataset)
        result_name = "{}_{}_result".format(result_dataset.name(), weight_name)
        result_dataset.set_name(result_name)
        result_dataset.write_data_to_json(result_name + ".json")
        print '--------------------------------------------------'


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Test on dataset")

    parser.add_argument("-w", "--weights_files", nargs='+', required=True, help="trained weights")
    parser.add_argument("-n", "--net_file", required=True, help="Deploy network")
    parser.add_argument("-d", "--dataset_file", required=True, help="Path to RenderAndCompare JSON dataset file")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu Id.")
    args = parser.parse_args()

    assert osp.exists(args.net_file), "Net filepath {} does not exist.".format(args.net_file)
    assert osp.exists(args.dataset_file), "Dataset filepath {} does not exist.".format(args.dataset_file)
    assert args.weights_files, "Weights files cannot be empty"

    for weights_file in args.weights_files:
        assert weights_file.endswith('.caffemodel'), "Weights file {} is nopt a valid Caffe weight file".format(weights_file)

    # sort the weight files
    args.weights_files.sort(key=lambda f: int(filter(str.isdigit, f)))

    print 'Loading dataset from {}'.format(args.dataset_file)
    dataset = Dataset.from_json(args.dataset_file)
    print 'Loaded {} dataset with {} annotations'.format(dataset.name(), dataset.num_of_annotations())

    print 'User provided {} weight files'.format(len(args.weights_files))
    test_all_weights_files(args.weights_files, args.net_file, dataset, args.gpu)


if __name__ == '__main__':
    main()
