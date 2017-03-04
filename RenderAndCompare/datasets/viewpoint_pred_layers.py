from datalayer import DataLayer
from image_loaders import BatchImageLoader
from random import shuffle
import numpy as np
import os.path as osp
import argparse
import caffe


class ViewpoinPredictionDataLayer(DataLayer):

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='View Prediction Data Layer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=[227, 227], type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[104.00698793, 116.66876762, 122.67891434], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- ViewpoinPredictionDataLayer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up ViewpoinPredictionDataLayer ..."
        assert len(top) >= 5, 'requires atleas one data and viewpoint tops' 

        # params is expected as argparse style string
        self.params = self.parse_param_str(self.param_str)

        # ----- Reshape tops -----#
        # data_shape = B x C x H x W
        # azimuth_shape = B x

        top[0].reshape(self.params.batch_size, 3, self.params.im_size[1], self.params.im_size[0])   #Image Data
        top[1].reshape(self.params.batch_size,) # AzimuthTarget
        top[2].reshape(self.params.batch_size,) # ElevationTarget
        top[3].reshape(self.params.batch_size,) # TiltTarget
        top[4].reshape(self.params.batch_size,) # distanceTarget

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(self.params.mean_bgr).reshape(1, 3, 1, 1)

        print "ViewpoinPredictionDataLayer has been setup."

    def set_dataset(self, dataset):
        assert dataset.num_of_annotations() >= self.params.batch_size, 'Numbers of data points ({}) should be >= batch size ({})'.format(
            len(dataset.num_of_annotations()), len(self.params.batch_size))

        image_files = []
        self.viewpoints = []

        for i in xrange(dataset.num_of_annotations()):
            annotation = dataset.annotations()[i]
            
            img_path= osp.join(dataset.rootdir(), annotation['image_file'])
            image_files.append(img_path)

            viewpoint = annotation['viewpoint']
            self.viewpoints.append(viewpoint)


        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(image_files, self.params.im_size)


        # set of data indices in [0, num_of_images)
        self.data_ids = range(len(image_files))
        self.curr_data_ids_idx = 0                  # current_data_id

        # Shuffle from the begining if in the train phase
        if (self.phase == caffe.TRAIN):
            shuffle(self.data_ids)

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

            self.curr_data_ids_idx += 1

        # subtarct mean from image data blob
        top[0].data[...] -= self.mean_bgr


class QuantizeViewPoint(caffe.Layer):
    """
    Converts continious ViewPoint measurement to quantized discrete labels
    Note the original viewpoint is asumed to lie in [0, 360)
    """
    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='Quantize Layer')
        parser.add_argument("-b", "--num_of_bins", default=24, type=int, help="Number of bins")
        params = parser.parse_args(param_str.split())

        print "------------- Quantize Layer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)
        self.degrees_per_bin = 360.0 / params.num_of_bins

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.floor_divide(bottom[0].data , self.degrees_per_bin)
    
    def backward(self, bottom, top):
        pass


class DeQuantizeViewPoint(caffe.Layer):
    """
    Converts quantized viewpoint labels to continious ViewPoint estimate
    Note the input is expected to lie within [0, num_of_bins)
    """
    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='Quantize Layer')
        parser.add_argument("-b", "--num_of_bins", default=24, type=int, help="Number of bins")
        params = parser.parse_args(param_str.split())

        print "------------- Quantize Layer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)
        self.degrees_per_bin = 360.0 / params.num_of_bins

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = (self.degrees_per_bin  * bottom[0].data) + self.degrees_per_bin / 2.0
    
    def backward(self, bottom, top):
        pass
