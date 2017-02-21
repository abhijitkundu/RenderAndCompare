#!/usr/bin/python
import os
import sys
import random

'''
@brief:
    extract labels from rendered images
@input:
    xxx/03790512_13a245da7b0567509c6d15186da929c5_a035_e009_t-01_d004.png
@output:
    (35,9,359)
'''
def path2label(path):
    parts = os.path.basename(path).split('_')
    azimuth = int(parts[-4][1:]) % 360
    elevation = int(parts[-3][1:]) % 360
    tilt = int(parts[-2][1:]) % 360
    return (azimuth, elevation, tilt)


'''
@brief:
    saves filenames and corresponmding label (azimuth, elevation, tilt)
@input:
    label_file  - label file to save to
    image_filename_label_pairs -  data to save
@output:
    save "<image_filepath> <azimuth> <elevation> <tilt>" to file.
'''
def save_image_filename_label_pairs(label_file, image_filename_label_pairs):
    fout = open(label_file, 'w')
    print "Writing %d image_filename_label_pairs to %s" % (len(image_filename_label_pairs), label_file)
    for filename_label in image_filename_label_pairs:
        label = filename_label[1]
        fout.write('%s %d %d %d\n' % (filename_label[0], label[0], label[1], label[2]));
    fout.close()


'''
@brief:
    get rendered image filenames and annotations, save to specified files.
@input:
    category_image_folder - containing mutiple folders of images with different shapes
    [train,test]_image_label_file - output file list filenames
    train_ratio - ratio of training images vs. all images
@output:
    save "<image_filepath> <azimuth> <elevation> <tilt>" to files.
'''
def get_one_category_image_label_file(category_image_folder, train_image_label_file, test_image_label_file, train_ratio = 0.9):
    shape_md5s = os.listdir(category_image_folder)

    image_filenames = []
    for k,md5 in enumerate(shape_md5s):
        shape_folder = os.path.join(category_image_folder, md5)
        shape_images = [os.path.join(shape_folder, x) for x in os.listdir(shape_folder)]
        image_filenames += shape_images

    image_filename_label_pairs = [(fpath,path2label(fpath)) for fpath in image_filenames]
    random.shuffle(image_filename_label_pairs)

    train_test_split = int(len(image_filename_label_pairs)*train_ratio)
    train_data = image_filename_label_pairs[0:train_test_split]
    test_data = image_filename_label_pairs[train_test_split:]

    save_image_filename_label_pairs(train_image_label_file, train_data)
    save_image_filename_label_pairs(test_image_label_file, test_data)


if __name__ == '__main__':

    g_shape_synset_name_pairs = [('02958343', 'car')]
    syn_images_cropped_bkg_overlaid_dir = '/media/Scratchspace/Render4CNNOutput/RenderedImages_voc2012/syn_images_cropped_bkg_overlaid'
    prefix = 'syn_voc2012_'
    train_ratio = 0.9

    # get image filenames and labels, separated to train/test sets
    for synset, name in g_shape_synset_name_pairs:
        category_image_folder = os.path.join(syn_images_cropped_bkg_overlaid_dir, synset)
        get_one_category_image_label_file(category_image_folder, prefix + name + '_train.txt', prefix + name + '_test.txt', train_ratio)
