from datalayer import AbstractDataLayer
from RenderAndCompare.datasets import BatchImageLoader
from random import shuffle
import numpy as np
import os.path as osp
import argparse
import caffe


class ArticulatedObjectDataLayer(AbstractDataLayer):

    def parse_param_str(self, param_str):
        top_names_choices = ['data', 'shape', 'pose']
        default_mean_bgr = [104.00698793, 116.66876762, 122.67891434]  # CaffeNet
        default_im_size = [227, 227]  # CaffeNet

        parser = argparse.ArgumentParser(description='Articulated Object Data Layer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=default_im_size, type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=default_mean_bgr, type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        parser.add_argument("-t", "--top_names", nargs='+', choices=top_names_choices, required=True, type=str, help="ordered list of top names e.g data shape")

        params = parser.parse_args(param_str.split())

        print "------------- ArticulatedObjectDataLayer Config -----------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up ArticulatedObjectDataLayer ..."

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        # Store the ordered list of top_names
        self.top_names = params.top_names
        # Store batch size as member variable for use in other methods
        self.batch_size = params.batch_size
        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(params.mean_bgr).reshape(1, 3, 1, 1)

        assert 'data' in self.top_names, 'Requires atleast data layer'
        assert len(top) == len(self.top_names), 'Number of tops do not match specified top_names'

        # Reshape image data top (B, C, H, W)
        if 'data' in self.top_names:
            top[self.top_names.index('data')].reshape(self.batch_size, 3, params.im_size[1], params.im_size[0])  # Image Data

        # Reshape shape top (B, 10)
        if 'shape' in self.top_names:
            top[self.top_names.index('shape')].reshape(self.batch_size, 10)

        # Reshape pose top (B, 10)
        if 'pose' in self.top_names:
            top[self.top_names.index('pose')].reshape(self.batch_size, 10)

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(params.im_size)

        if 'shape' in self.top_names:
            self.shape_params = []

        if 'pose' in self.top_names:
            self.pose_params = []

        print 'ArticulatedObjectDataLayer has been setup.'

    def add_dataset(self, dataset):
        print '---- Adding data from {} datatset -----'.format(dataset.name())

        image_files = []
        cropping_boxes = []
        for i in xrange(dataset.num_of_annotations()):
            annotation = dataset.annotations()[i]
            img_path = osp.join(dataset.rootdir(), annotation['image_file'])
            image_files.append(img_path)
            visible_bbx = np.array(annotation['visible_bbx'], dtype=np.float)  # gt box (only visible path)
            # We need to probably do jittering
            cropping_boxes.append(visible_bbx)

            if hasattr(self, 'shape_params'):
                self.shape_params.append(np.array(annotation['body_shape'], dtype=np.float))

            if hasattr(self, 'pose_params'):
                self.shape_params.append(np.array(annotation['body_pose'], dtype=np.float))

        # self.image_loader.crop_and_preload_images(image_files, cropping_boxes)
        self.image_loader.preload_images(image_files)
        print "--------------------------------------------------------------------"

    def generate_datum_ids(self):
        num_of_data_points = len(self.image_loader)

        for attr in ['viewpoints', 'shape_params', 'pose_params']:
            if hasattr(self, attr):
                assert len(getattr(self, attr)) == num_of_data_points

        # set of data indices in [0, num_of_data_points)
        self.data_ids = range(num_of_data_points)
        self.curr_data_ids_idx = 0

        # Shuffle from the begining if in the train phase
        if (self.phase == caffe.TRAIN):
            shuffle(self.data_ids)

        print 'Total number of data points (annotations) = {:,}'.format(num_of_data_points)
        return num_of_data_points

    def forward(self, bottom, top):
        """
        Load current batch of data and labels to caffe blobs
        """

        assert hasattr(self, 'data_ids'), 'Most likely data has not been initialized before calling forward()'
        assert len(self.data_ids) > self.batch_size, 'batch size cannot be smaller than total number of data points'

        # For Debug
        # print "{} -- {}".format(self.data_ids[self.curr_data_ids_idx],
        # self.data_ids[self.curr_data_ids_idx + 100])

        for i in xrange(self.batch_size):
            # Did we finish an epoch?
            if self.curr_data_ids_idx == len(self.data_ids):
                self.curr_data_ids_idx = 0
                shuffle(self.data_ids)

            data_idx = self.data_ids[self.curr_data_ids_idx]

            if 'data' in self.top_names:
                top[self.top_names.index('data')].data[i, ...] = self.image_loader[data_idx]

            if hasattr(self, 'shape_params'):
                top[self.top_names.index('shape')].data[i, ...] = self.shape_params[data_idx]

            if hasattr(self, 'pose_params'):
                top[self.top_names.index('pose')].data[i, ...] = self.pose_params[data_idx]

            self.curr_data_ids_idx += 1

        # subtarct mean from full image data blob
        if 'data' in self.top_names:
            top[self.top_names.index('data')].data[...] -= self.mean_bgr
