#!/usr/bin/env python

import _init_paths
import os.path as osp
import RenderAndCompare as rac
import numpy as np
import cv2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualizes annotations')
    parser.add_argument('annotation_file', help='Path to our json Annotation file')
    args = parser.parse_args()


    dataset = rac.datasets.Dataset.from_json(args.annotation_file)
    print dataset

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('bbox', cv2.WINDOW_AUTOSIZE)

    for i in xrange(dataset.num_of_annotations()):
        annotation = dataset.annotations()[i]

        img_path= osp.join(dataset.rootdir(), annotation['image_file'])
        assert osp.exists(img_path), 'Image file {} does not exist'.format(img_path)
        image =cv2.imread(img_path)

        viewpoint = annotation['viewpoint']
        print '-a {} -e{} -t {} -d {}'.format(viewpoint[0], viewpoint[1], viewpoint[2], viewpoint[3])

        bbx_amodal = rac.geometry.BoundingBox.fromRect(annotation['bbx_amodal']) 
        bbx_crop = rac.geometry.BoundingBox.fromRect(annotation['bbx_crop'])

        bbx_image = np.zeros((540, 960, 3),dtype=np.uint8)
        cv2.rectangle(bbx_image,
            tuple(np.floor(bbx_amodal.min()).astype(int)),
            tuple(np.ceil(bbx_amodal.max()).astype(int)),
            (0,255,0),1)

        cv2.rectangle(bbx_image,
            tuple(np.floor(bbx_crop.min()).astype(int)),
            tuple(np.ceil(bbx_crop.max()).astype(int)),
            (0,0,255),1)

        

        cv2.imshow('image',image)
        cv2.imshow('bbox',bbx_image)
        key = cv2.waitKey(0)

        if key == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            break




