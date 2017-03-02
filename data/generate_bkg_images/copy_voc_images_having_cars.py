import os.path as osp
import pandas as pd
from shutil import copyfile

voc_root_dir = '/media/Scratchspace/PascalVOC-Datasets/VOCdevkit2007/VOC2007'
voc_images_dir = osp.join(voc_root_dir, 'JPEGImages')
image_set_file = osp.join(voc_root_dir, 'ImageSets', 'Main', 'car_train.txt')

df = pd.read_csv(image_set_file, delim_whitespace=True, names = ['image_name', 'presence'], dtype={'image_name': object, 'presence': int}, header=None)
car_images = df.loc[df['presence'] >= 0]
car_image_names = car_images['image_name'].tolist()

for image_name in car_image_names:
    image_filename = image_name + '.jpg'
    copyfile(osp.join(voc_images_dir, image_filename), image_filename)

