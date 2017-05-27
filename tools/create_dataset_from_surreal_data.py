#!/usr/bin/env python

import _init_paths
import os.path as osp
from os import walk
import fnmatch
import h5py
import RenderAndCompare as rac
import numpy as np
from collections import OrderedDict
from tqdm import tqdm


def create_annotation_for_single_image(image_file, root_dir, pose_mean=None, pose_basis=None):
    splits_ = image_file.split(osp.sep)
    frame_name = osp.splitext(splits_[-1])[0]

    annotation = OrderedDict()
    annotation['image_file'] = osp.relpath(image_file, root_dir)

    segm_dir = osp.join(osp.dirname(image_file), '../segm')
    assert osp.exists(segm_dir), 'Segm directory "{}" does not exist'.format(segm_dir)

    segm_file = osp.join(segm_dir, frame_name + '.png')
    assert osp.exists(segm_file), 'segm file "{}" does not exist'.format(segm_file)
    annotation['segm_file'] = osp.relpath(segm_file, root_dir)

    info_dir = osp.join(osp.dirname(image_file), '../info_neutral_bbx')
    assert osp.exists(info_dir), 'Info directory "{}" does not exist'.format(info_dir)

    info_file = osp.join(info_dir, frame_name + '.h5')
    assert osp.exists(info_file), 'Info file "{}" does not exist'.format(info_file)

    with h5py.File(info_file, 'r') as hf:
        assert hf['gender'][...] == 'neutral', 'Only supports gender neutral params'
        annotation['bbx_visible'] = np.squeeze(hf['visible_bbx'][...]).tolist()
        annotation['camera_extrinsic'] = hf['camera_extrinsics'][...].flatten().tolist()
        annotation['model_pose'] = hf['model_pose'][...].flatten().tolist()
        annotation['shape_param'] = hf['body_shape'][...].astype(np.float).tolist()
        body_pose = hf['body_pose'][...]
        assert body_pose.shape == (69,)

    if pose_mean is not None and pose_basis is not None:
        encoded_body_pose = pose_basis.T.dot(body_pose - pose_mean)
        assert encoded_body_pose.shape == (pose_basis.shape[1],), 'unexpected encoded_body_pose.shape = {}'.format(encoded_body_pose.shape)
        annotation['pose_param'] = encoded_body_pose.astype(np.float).tolist()
    else:
        annotation['pose_param'] = body_pose.astype(np.float).tolist()

    return annotation


if __name__ == '__main__':
    smpl_data_dir = osp.join(_init_paths.root_dir, 'data', 'smpl_data')
    assert osp.exists(smpl_data_dir), 'SMPL Data directory "{}" does not exist'.format(smpl_data_dir)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", required=True, help="Input folder containing single image surreal data")
    parser.add_argument("-d", "--dataset_name", default='surreal_xxxxx', help="Dataset name")
    parser.add_argument("-m", "--smpl_model_file", default=osp.join(smpl_data_dir, 'smpl_neutral_lbs_10_207_0.h5'), help="Path to smpl model file")
    parser.add_argument("-p", "--pose_pca_file", help="Pose PCA file. If not provided use full 69 params")
    parser.add_argument("-r", "--remove_empty_bbx", default=1, type=int, help="Remove empty box")
    args = parser.parse_args()

    assert osp.exists(args.input_folder), 'Input data folder "{}" does not exist'.format(args.input_folder)
    assert osp.isdir(args.input_folder), 'Input data folder "{}" is not a directory'.format(args.input_folder)
    assert osp.exists(args.smpl_model_file), 'SMPL model file "{}" does not exist'.format(args.smpl_model_file)

    print "------------- Config ------------------"
    for arg in vars(args):
        print "{}\t= {}".format(arg, getattr(args, arg))

    print 'Searching for images inside {}'.format(args.input_folder)
    image_files = []
    for root, dirnames, filenames in walk(args.input_folder, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in ['segm', 'info']]
        for filename in fnmatch.filter(filenames, '*.png'):
            image_files.append(osp.join(root, filename))
    print 'Found {:,} images in {}'.format(len(image_files), args.input_folder)

    dataset = rac.datasets.Dataset(args.dataset_name)
    dataset.set_rootdir(args.input_folder)

    metainfo = {}
    metainfo['shape_param_dimension'] = 10

    if args.pose_pca_file is not None:
        print 'Loading smpl pose pca basis from {}'.format(args.pose_pca_file)
        with h5py.File(args.pose_pca_file, 'r') as hf:
            pose_mean = np.squeeze(hf['pose_mean'][...])
            pose_basis = hf['pose_basis'][...]
            assert pose_mean.shape == (69,)
            assert pose_basis.shape[0] == 69
        metainfo['pose_param_dimension'] = pose_basis.shape[1]
    else:
        print 'No pose pca file provided. Will not do PCA on pose.'
        metainfo['pose_param_dimension'] = 69

    dataset.set_metainfo(metainfo)

    print 'Creating annotations for {:,} images'.format(len(image_files))
    for image_file in tqdm(image_files):
        if args.pose_pca_file is not None:
            annotation = create_annotation_for_single_image(image_file, dataset.rootdir(), pose_mean, pose_basis)
        else:
            annotation = create_annotation_for_single_image(image_file, dataset.rootdir())
        # check the bbx and add only if its a good one
        bbx = np.array(annotation['bbx_visible']).astype(np.int)

        if args.remove_empty_bbx == 1:
            if (bbx[3] - bbx[1]) <= 0 or (bbx[2] - bbx[0]) <= 0:
                tqdm.write('Invalid bbx = {} in image {}'.format(bbx, image_file))
                continue
        dataset.add_annotation(annotation)

    print 'Finished creating dataset with {} annotations from {} images'.format(dataset.num_of_annotations(), len(image_files))

    output_json_file = args.dataset_name + '.json'
    print 'Saving JSON dataset file at {}'.format(output_json_file)
    dataset.write_data_to_json(output_json_file)
