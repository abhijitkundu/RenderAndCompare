#!/usr/bin/env python

import _init_paths
import os.path as osp
import RenderAndCompare as rac
import numpy as np
import cv2


def visualize_dataset(dataset):
    cv2.startWindowThread()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for i in xrange(dataset.num_of_annotations()):
        annotation = dataset.annotations()[i]

        img_path = osp.join(dataset.rootdir(), annotation['image_file'])
        assert osp.exists(img_path), 'Image file {} does not exist'.format(img_path)
        image = cv2.imread(img_path)

        bbx = np.array(annotation['bbx_visible']).astype(np.int)

        cv2.rectangle(image,
                      (bbx[0], bbx[1]),
                      (bbx[2] - 1, bbx[3] - 1),
                      (0, 255, 0), 1)

        cv2.imshow('image', image)
        key = cv2.waitKey(args.pause)

        if (bbx[3] - bbx[1]) < 0 or (bbx[2] - bbx[0]) < 0:
            print 'Invalid bbx = {}'.format(bbx)
            key = cv2.waitKey(0)

        if key == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualizes annotations')
    parser.add_argument('annotation_file', help='Path to our json Annotation file')
    parser.add_argument("-p", "--pause", default=0, type=int, help="Set number of milliseconds to pause. Use 0 to pause indefinitely")
    args = parser.parse_args()

    print 'Parsing annotation {} ...'.format(args.annotation_file)
    dataset = rac.datasets.Dataset.from_json(args.annotation_file)
    print dataset

    visualize_dataset(dataset)
