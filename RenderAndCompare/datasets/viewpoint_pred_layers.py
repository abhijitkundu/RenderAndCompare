import caffe

import numpy as np
import os.path as osp
import argparse
from random import shuffle
from image_loaders import BatchImageLoader


class ViewpoinPredictionDataLayer(caffe.Layer):

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='View Prediction Data Layer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-s", "--split_file", default='../../data/render4cnn/car_train.txt', help="Path to split file")
        parser.add_argument("-a", "--azimuth_bins", default=24, type=int, help="Number of bins for azimuth")
        parser.add_argument("-wh", "--im_size", nargs=2, default=[227, 227], type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[104.00698793, 116.66876762, 122.67891434], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- ViewpoinPredictionDataLayer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        # store input as class variables
        self.batch_size = params.batch_size

        # ----- Reshape tops -----#
        # data_shape = B x C x H x W
        # label_shape = B x

        top[0].reshape(self.batch_size, 3, params.im_size[
                       1], params.im_size[0])
        top[1].reshape(self.batch_size,)

        image_files, self.labels = self.get_image_files_and_labels(params)
        assert len(image_files) == len(self.labels), 'Numbers of image files ({}) do not match number of labels ({})'.format(
            len(image_files), len(self.labels))
        assert len(image_files) >= self.batch_size, 'Numbers of image files ({}) should be >= batch size ({})'.format(
            len(image_files), len(self.batch_size))

        # set of data indices in [0, num_of_images)
        self.data_ids = range(len(image_files))
        self.curr_data_ids_idx = 0                  # current_data_id

        # Shuffle from the begining if in the train phase
        if (self.phase == caffe.TRAIN):
            shuffle(self.data_ids)

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(image_files, params.im_size)

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(params.mean_bgr).reshape(1, 3, 1, 1)

        print "ViewpoinPredictionDataLayer initialized for split: {}.".format(osp.basename(params.split_file))

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def forward(self, bottom, top):
        """
        Load current batch of data and labels to caffe blobs
        """
        # For Debug
        # print "{} -- {}".format(self.data_ids[self.curr_data_ids_idx],
        # self.data_ids[self.curr_data_ids_idx + 100])

        for i in xrange(self.batch_size):
            # Did we finish an epoch?
            if self.curr_data_ids_idx == len(self.data_ids):
                self.curr_data_ids_idx = 0
                shuffle(self.data_ids)

            # Add directly to the caffe data layer
            data_idx = self.data_ids[self.curr_data_ids_idx]
            top[0].data[i, ...] = self.image_loader[data_idx]
            top[1].data[i, ...] = self.labels[data_idx]

            self.curr_data_ids_idx += 1

        # subtarct mean from image data blob
        top[0].data[...] -= self.mean_bgr

    def backward(self, bottom, top):
        """
        These layers does not back propagate
        """
        pass

    def get_image_files_and_labels(self, params):
        split_file = params.split_file
        assert osp.exists(
            split_file), 'Path does not exist: {}'.format(split_file)

        azimuth_bins = params.azimuth_bins
        degrees_per_azimuth_bin = 360 / azimuth_bins

        image_files = []
        labels = []
        with open(split_file, 'rb') as f:
            for line in f:
                line = line.rstrip('\n')
                [filename, azimuth, elevation, tilt] = line.split(' ')
                azimuth = int(azimuth)
                azimuth_label = azimuth / degrees_per_azimuth_bin
                assert azimuth_label < azimuth_bins, "label (%d) >= num_classes (%d) for Image at %s" % (
                    azimuth_label, azimuth_bins, osp.basename(filename))
                image_files.append(filename)
                labels.append(azimuth_label)

        return image_files, labels

    def make_rgb8_from_blob(self, blob_data):
        assert blob_data.ndim == 3, 'expects a color image (dim: 3)'
        image = blob_data + self.mean_bgr.reshape(3, 1, 1)
        image = image.transpose(1, 2, 0)
        image = image[:, :, ::-1]  # change to RGB
        return np.uint8(image)
