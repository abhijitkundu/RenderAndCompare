#!/usr/bin/env python
"""This script creates split files for pascal3d imagenet images"""

from glob import glob
import os.path as osp

pascal3d_rooot_dir = osp.join(osp.dirname(__file__), 'Pascal3D-Dataset')

val_split_files = glob(osp.join(pascal3d_rooot_dir, 'Image_sets', '*_imagenet_val.txt'))
print "We have {} val split_files".format(len(val_split_files))
val_image_names = set()
for val_split_file in val_split_files:
    val_image_names.update([x.rstrip() for x in open(val_split_file)])
print "We have {} images in val set".format(len(val_image_names))


train_split_files = glob(osp.join(pascal3d_rooot_dir, 'Image_sets', '*_imagenet_train.txt'))
print "We have {} train split_files".format(len(train_split_files))
train_image_names = set()
for train_split_file in train_split_files:
    train_image_names.update([x.rstrip() for x in open(train_split_file)])
print "We have {} images in train set".format(len(train_image_names))

trainval_image_names = val_image_names | train_image_names
assert len(trainval_image_names) == len(val_image_names) + len(train_image_names)

with open('imagenet_val.txt', mode='wt') as myfile:
    myfile.write('\n'.join(val_image_names))
with open('imagenet_train.txt', mode='wt') as myfile:
    myfile.write('\n'.join(train_image_names))
with open('imagenet_trainval.txt', mode='wt') as myfile:
    myfile.write('\n'.join(trainval_image_names))
