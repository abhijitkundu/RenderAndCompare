from datalayer import DataLayer
from RenderAndCompare.datasets import BatchImageLoader
from random import shuffle
import numpy as np
import os.path as osp
import argparse
import caffe


class CropPredictionDataLayer(DataLayer):

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='Crop Prediction Data Layer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=[227, 227], type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[104.00698793, 116.66876762, 122.67891434], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- CropPredictionDataLayer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up CropPredictionDataLayer ..."
        assert len(top) == 3, 'requires 3 tops: (image) data, bbx_amoda, and bbx_crop'

        # params is expected as argparse style string
        self.params = self.parse_param_str(self.param_str)

        # ----- Reshape tops -----#
        # data_shape = B x C x H x W
        # azimuth_shape = B x

        top[0].reshape(self.params.batch_size, 3, self.params.im_size[1], self.params.im_size[0])  # Image Data
        top[1].reshape(self.params.batch_size, 4)  # AmodalBbxTarget
        top[2].reshape(self.params.batch_size, 4)  # CropBbxTarget

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(self.params.mean_bgr).reshape(1, 3, 1, 1)

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(self.params.im_size)

        # Create placeholder for GT annotations
        self.amodal_boxes = []
        self.crop_boxes = []

        print 'CropPredictionDataLayer has been setup.'

    def add_dataset(self, dataset):
        print '---- Adding data from {} datatset -----'.format(dataset.name())

        image_files = []
        for i in xrange(dataset.num_of_annotations()):
            annotation = dataset.annotations()[i]
            img_path = osp.join(dataset.rootdir(), annotation['image_file'])
            image_files.append(img_path)

            self.amodal_boxes.append(np.array(annotation['bbx_amodal'], dtype=np.float))
            self.crop_boxes.append(np.array(annotation['bbx_crop'], dtype=np.float))

        self.image_loader.preload_images(image_files)
        print "--------------------------------------------------------------------"

    def generate_datum_ids(self):
        assert len(self.amodal_boxes) == len(self.crop_boxes)
        num_of_data_points = len(self.amodal_boxes)

        # set of data indices in [0, num_of_data_points)
        self.data_ids = range(num_of_data_points)
        self.curr_data_ids_idx = 0

        # Shuffle from the begining if in the train phase
        if (self.phase == caffe.TRAIN):
            shuffle(self.data_ids)

        print 'Total number of data points (annotations) = {:,}'.format(num_of_data_points)

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

            top[1].data[i, ...] = self.amodal_boxes[data_idx]
            top[2].data[i, ...] = self.crop_boxes[data_idx]

            self.curr_data_ids_idx += 1

        # subtarct mean from image data blob
        top[0].data[...] -= self.mean_bgr


class GenerateCropTransformationTargets(caffe.Layer):
    """
    Computes the relative bbx transfromation  fromtwo bounding boxes
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers box_a and box_c'
        assert len(top) == 1, 'requires a single layer.top'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0]
        assert bottom[0].data.shape[1] == 4
        assert bottom[1].data.shape[1] == 4
        top[0].reshape(bottom[0].data.shape[0], 4)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        tfm = np.zeros((bottom[0].data.shape[0], 4))
        tfm[:, :2] = (bottom[1].data[:, :2] - bottom[0].data[:, :2]) / bottom[0].data[:, 2:]
        tfm[:, 2:] = np.log(bottom[1].data[:, 2:] / bottom[0].data[:, 2:])
        top[0].data[...] = tfm

    def backward(self, top, propagate_down, bottom):
        pass
