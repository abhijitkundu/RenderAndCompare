#!/usr/bin/env python

import os.path as osp
import sys
from math import floor
import cv2
import random

coco_dir = osp.join( osp.dirname(__file__), 'coco')
sys.path.insert(0, osp.join(coco_dir, 'PythonAPI'))

from pycocotools.coco import COCO


def doBoxesIntersect(a, b):
    return ((min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])) >=0 ) and ((min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])) >=0 )

def drawBbxOnImage(image, box, color=(0,0,255)):
    l = int(floor(box[0]))
    t = int(floor(box[1]))

    r = int(floor(box[0] + box[2]))
    b = int(floor(box[1] + box[3]))

    cv2.rectangle(image,(l, t),(r,b),color,3)

def get_cropped_image(image, bbx):
    return image[bbx[1]:(bbx[1]+bbx[3]), bbx[0]:(bbx[0]+bbx[2])]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make Backgrounf images from COCO')
    parser.add_argument("-s", "--save", default=0, type=int, help="Set save mode")
    parser.add_argument("-v", "--visualize", default=1, type=int, help="Set visualization mode (0 to turn off)")
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
   
    args = parser.parse_args()

    visualize = (args.visualize != 0)
    save = (args.save != 0)

    dataType ='train2014'
    annFile ='%s/annotations/instances_%s.json'%(coco_dir,dataType)

    # initialize COCO api for instance annotations
    coco=COCO(annFile)



    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print 'COCO categories: \n\n', ' '.join(nms)

    nms = set([cat['supercategory'] for cat in cats])
    print 'COCO supercategories: \n', ' '.join(nms)


    excludeCatIds = coco.getCatIds(supNms=['food','indoor','appliance','animal','furniture','electronic','kitchen']);
    # includeCatIds = coco.getCatIds(supNms=['person','vehicle','outdoor']);
    includeCatIds = coco.getCatIds(catNms=['bicycle','car','motorcycle','bus','train','truck','traffic light','fire hydrant','stop sign','parking meter']);

    imgIdsSet = set()

    print "Adding images from Include categories"
    for catId in includeCatIds:
        imgIds = coco.getImgIds(catIds=catId);
        print "Adding %d images with %s" % (len(imgIds), coco.loadCats(catId)[0]['name'])
        for imgId in imgIds:
            imgIdsSet.add(imgId)
            
    print "Removing images from Exclude categories"
    for catId in excludeCatIds:
        imgIds = coco.getImgIds(catIds=catId);
        print "Removing %d images with %s" % (len(imgIds), coco.loadCats(catId)[0]['name'])
        for imgId in imgIds:
            imgIdsSet.discard(imgId)
        
    print "Total Found %d images" % (len(imgIdsSet))
    imgIds = list(imgIdsSet)


    num_of_trails = 10000
    min_width = 280
    min_height = 224

    if visualize is True:
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    for img_id in imgIds:
        img_info = coco.loadImgs(img_id)[0]
        image =cv2.imread('%s/images/%s/%s'%(coco_dir,dataType,img_info['file_name']))

        iw = image.shape[1]
        ih = image.shape[0]

        if iw <= min_width or ih <= min_height:
            continue

        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        gtbboxes = [ann['bbox'] for ann in anns]


        best_bbx = [0, 0, 0, 0]
        best_bbx_area = 0


        for t in xrange(num_of_trails):
            w = random.randint(min_width, iw)
            h = random.randint(min_height, ih)

            x = random.randint(0, iw - w)
            y = random.randint(0, ih - h)

            bbx = [ x, y, w, h]

            valid_bbx = True
            for gtbox in gtbboxes:
                if doBoxesIntersect(bbx, gtbox) is True:
                    valid_bbx = False
                    break

            if valid_bbx is True:
                bbx_area = bbx[2] * bbx[3]
                if bbx_area > best_bbx_area:
                    best_bbx_area = bbx_area
                    best_bbx = bbx

        if save is True and best_bbx_area > 0:
            cropped_bkg_image = get_cropped_image(image, best_bbx)
            image_filename_stem = osp.splitext(osp.basename(img_info['file_name']))[0]
            save_image_file_name = '%s_cropped_%d_%d_%d_%d.jpg' % (image_filename_stem, best_bbx[0], best_bbx[1], best_bbx[2], best_bbx[3])
            cv2.imwrite(save_image_file_name, cropped_bkg_image)
            print 'Wrote cropped bkg image to %s' % save_image_file_name



        if visualize is True:
            for box in gtbboxes:
                drawBbxOnImage(image, box)

            if best_bbx_area > 0:
                drawBbxOnImage(image, best_bbx, (0, 255, 0))

            cv2.imshow('image',image)
            key = cv2.waitKey(args.pause)

            if key == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()
                break