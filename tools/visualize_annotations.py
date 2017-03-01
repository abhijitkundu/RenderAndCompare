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
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
    parser.add_argument("-wh", "--camera_size", nargs=2, default=[960, 540], type=int, metavar=('WIDTH', 'HEIGHT'), help="Camera Image Size [width, height]")
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
        print '{} -a {} -e {} -t {} -d {}'.format(annotation['image_file'], viewpoint[0], viewpoint[1], viewpoint[2], viewpoint[3])

        assert all(i >= 0 for i in viewpoint), 'Bad viewpoint {}'.format(viewpoint)
        assert all(i < 360 for i in viewpoint[:3]), 'Bad viewpoint {}'.format(viewpoint)

        bbx_amodal = rac.geometry.BoundingBox.fromRect(annotation['bbx_amodal']) 
        bbx_crop = rac.geometry.BoundingBox.fromRect(annotation['bbx_crop'])

        image_size = np.array([image.shape[1], image.shape[0]])
        assert (image_size == bbx_crop.size()).all(), 'Image size {} and bbx_crop size {} should match'.format(image_size, bbx_crop.size())


        camera_width = args.camera_size[0]
        camera_height = args.camera_size[1]
        principal_point = np.array([camera_width/2.0, camera_height/2.0], dtype=np.float)

        bbx_amodal.translate(principal_point)
        bbx_crop.translate(principal_point)

        bbx_image = np.zeros((camera_height, camera_width, 3),dtype=np.uint8)
        bbx_image[bbx_crop.min()[1]:bbx_crop.max()[1],bbx_crop.min()[0]:bbx_crop.max()[0]] = image


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
        key = cv2.waitKey(args.pause)

        if key == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            break




