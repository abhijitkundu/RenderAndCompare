#!/usr/bin/env python

import _init_paths
from RenderAndCompare.datasets import ImageDataset
import numpy as np
import os.path as osp
from scipy import io as sio

def main():
    image_dataset_file = '/home/abhijit/Workspace/RenderAndCompare/experiments/vp_abbx_cp/Resnet50_FPN_vp96T_cpL1_abbxL1/Resnet50_fbn_FPN_roisF16_vp96T_cpL1_abbxL1_synth_pascal3d_kitti_3dvp/voc_2012_val_car_Resnet50_fbn_FPN_roisF16_vp96T_cpL1_abbxL1_synth_pascal3d_kitti_3dvp_iter_80000_result.json'
    print('Loading image dataset from {}'.format(image_dataset_file))
    image_datset = ImageDataset.from_json(image_dataset_file)
    print(image_datset)

    val_split = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
    val_image_names = open(val_split).read().splitlines()
    print(len(val_image_names))

    dets = [np.array([], dtype=np.float32)] * len(val_image_names)

    for im_info in image_datset.image_infos():
        num_of_objects = len(im_info['object_infos'])
        curr_image_dets = np.empty([num_of_objects, 5], dtype=np.float32)
        for i in xrange(num_of_objects):
            ob_info = im_info['object_infos'][i]
            curr_image_dets[i, :4] = ob_info['bbx_visible']
            curr_image_dets[i, 4] = ob_info['score']
        image_name = osp.splitext(osp.basename(im_info['image_file']))[0]

        dets[val_image_names.index(image_name)] = curr_image_dets
    
    sio.savemat('car_dets.mat', {'boxes': dets})

    with open('car_pred_view.txt', 'w') as f:
        for im_info in image_datset.image_infos():
            for ob_info in im_info['object_infos']:
                vp = np.degrees(ob_info['viewpoint']) % 360.0
                assert vp.shape == (3,)
                assert (vp >= 0.0).all() and (vp < 360.0).all(), "Expects viewpoint to be in [0, 360), but got {}".format(vp)
                f.write("{} {} {}\n".format(vp[0], vp[1], vp[2]))

if __name__ == '__main__':
    main()
