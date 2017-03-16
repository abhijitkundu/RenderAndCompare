from RenderAndCompare.datasets import BatchImageLoader
from random import shuffle
import os.path as osp
import numpy as np
import caffe
import argparse


class AbstractDataLayer(caffe.Layer):
    """
    Generic Data layer for caffe
    """

    def parse_param_str(self, param_str):
        """
        parse the param string passed from prtotxt file. Reimplement this if rerquired.
        """
        parser = argparse.ArgumentParser(description='AbstractDataLayer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=[227, 227], type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[104.00698793, 116.66876762, 122.67891434], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- AbstractDataLayer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        # params is expected as argparse style string
        self.params = self.parse_param_str(self.param_str)

        # ----- Reshape tops -----#
        # data_shape = B x C x H x W
        # label_shape = B x

        top[0].reshape(self.params.batch_size, 3, self.params.im_size[1], self.params.im_size[0])
        top[1].reshape(self.params.batch_size,)

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(self.params.mean_bgr).reshape(1, 3, 1, 1)

        print "AbstractDataLayer has been setup."

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def forward(self, bottom, top):
        """
        Reimplement this
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

    def make_rgb8_from_blob(self, blob_data):
        """
        conveneince method to get an rgb8 image (compatible for Matplotlib display)
        """
        assert blob_data.ndim == 3, 'expects a color image (dim: 3)'
        image = blob_data + self.mean_bgr.reshape(3, 1, 1)
        image = image.transpose(1, 2, 0)
        image = image[:, :, ::-1]  # change to RGB
        return np.uint8(image)

    def make_bgr8_from_blob(self, blob_data):
        """
        conveneince method to get an bgr8 image (compatible for OpenCV display)
        """
        assert blob_data.ndim == 3, 'expects a color image (dim: 3)'
        image = blob_data + self.mean_bgr.reshape(3, 1, 1)
        image = image.transpose(1, 2, 0)
        return np.uint8(image)


class DataLayer(AbstractDataLayer):

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='Data Layer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=[227, 227], type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[104.00698793, 116.66876762, 122.67891434], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "-------------------- DataLayer Config ----------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up DataLayer ..."
        assert len(top) >= 7, 'requires atleas one data, viewpoint tops, and box tops'

        # params is expected as argparse style string
        self.params = self.parse_param_str(self.param_str)

        # ----- Reshape tops -----#
        # data_shape = B x C x H x W
        # azimuth_shape = B x

        top[0].reshape(self.params.batch_size, 3, self.params.im_size[1], self.params.im_size[0])  # Image Data

        # ------------ viewpoint tops --------------- #
        top[1].reshape(self.params.batch_size,)  # AzimuthTarget
        top[2].reshape(self.params.batch_size,)  # ElevationTarget
        top[3].reshape(self.params.batch_size,)  # TiltTarget
        top[4].reshape(self.params.batch_size,)  # distanceTarget

        # --------- amodal and crop boxes ------------ #
        top[5].reshape(self.params.batch_size, 4)  # AmodalBbxTarget
        top[6].reshape(self.params.batch_size, 4)  # CropBbxTarget

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(self.params.mean_bgr).reshape(1, 3, 1, 1)

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(self.params.im_size)

        # Create placeholder for GT annotations
        self.viewpoints = []
        self.amodal_boxes = []
        self.crop_boxes = []

        print 'DataLayer has been setup.'

    def add_dataset(self, dataset):
        print '---- Adding data from {} datatset -----'.format(dataset.name())

        image_files = []
        for i in xrange(dataset.num_of_annotations()):
            annotation = dataset.annotations()[i]
            img_path = osp.join(dataset.rootdir(), annotation['image_file'])
            image_files.append(img_path)

            viewpoint = annotation['viewpoint']
            self.viewpoints.append(viewpoint)
            self.amodal_boxes.append(np.array(annotation['bbx_amodal'], dtype=np.float))
            self.crop_boxes.append(np.array(annotation['bbx_crop'], dtype=np.float))

        self.image_loader.preload_images(image_files)
        print "--------------------------------------------------------------------"

    def generate_datum_ids(self):
        assert len(self.amodal_boxes) == len(self.crop_boxes)
        assert len(self.amodal_boxes) == len(self.viewpoints)
        num_of_data_points = len(self.viewpoints)

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
        assert len(self.data_ids) > self.params.batch_size, 'batch size cannot be smaller than total number of data points'

        # For Debug
        # print "{} -- {}".format(self.data_ids[self.curr_data_ids_idx],
        # self.data_ids[self.curr_data_ids_idx + 100])

        for i in xrange(self.params.batch_size):
            # Did we finish an epoch?
            if self.curr_data_ids_idx == len(self.data_ids):
                self.curr_data_ids_idx = 0
                shuffle(self.data_ids)

            # Add directly to the caffe data layer
            data_idx = self.data_ids[self.curr_data_ids_idx]
            top[0].data[i, ...] = self.image_loader[data_idx]

            viewpoint = self.viewpoints[data_idx]
            top[1].data[i, ...] = viewpoint[0]
            top[2].data[i, ...] = viewpoint[1]
            top[3].data[i, ...] = viewpoint[2]
            top[4].data[i, ...] = viewpoint[3]

            top[5].data[i, ...] = self.amodal_boxes[data_idx]
            top[6].data[i, ...] = self.crop_boxes[data_idx]

            self.curr_data_ids_idx += 1

        # subtarct mean from image data blob
        top[0].data[...] -= self.mean_bgr
