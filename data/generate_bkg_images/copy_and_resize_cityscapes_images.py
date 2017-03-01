import os.path as osp
import glob
import cv2
from progressbar import ProgressBar, Percentage, Bar

images_dir = '/media/Scratchspace/Cityscapes-Dataset/leftImg8bit/train'
glob_pattern = images_dir + '/*/*.png'
out_dir = '/media/Scratchspace/BackGroundImages/cityscapes_train_resized'

filelist = glob.glob(glob_pattern)
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(filelist)).start()

for i, file in enumerate(filelist):
    image =cv2.imread(file)
    resized_image = cv2.resize(image, (960, 480))
    out_filename = osp.join(out_dir, osp.basename(file))
    cv2.imwrite(out_filename, resized_image)
    pbar.update(i+1)
pbar.finish()

