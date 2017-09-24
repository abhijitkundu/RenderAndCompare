import argparse
import os.path as osp
from random import shuffle, random

import numpy as np

import caffe
from RenderAndCompare.datasets import BatchImageLoader, crop_and_resize_image, uniform_crop_and_resize_image


class AbstractDataLayer(caffe.Layer):
    """
    Generic python Data layer for caffe
    """

    def parse_param_str(self, param_str):
        """
        parse the param string passed from prtotxt file. Reimplement this if rerquired.
        """
        parser = argparse.ArgumentParser(description='AbstractDataLayer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084],
                            type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- AbstractDataLayer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        """Reimplement Layer setup in this method"""
        # params is expected as argparse style string
        self.params = self.parse_param_str(self.param_str)

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(self.params.mean_bgr).reshape(1, 3, 1, 1)

        print "AbstractDataLayer has been setup."
        pass

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

    def generate_datum_ids(self):
        """Reimplement this"""
        pass

    def number_of_datapoints(self):
        """Number of data points (samples)"""
        if hasattr(self, 'data_ids'):
            return len(self.data_ids)
        return 0

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
    """Data Layer RCNN style. Inherits AbstractDataLayer"""

    def parse_param_str(self, param_str):
        top_names_choices = ['input_image', 'viewpoint', 'bbx_amodal', 'bbx_crop', 'center_proj']
        default_mean_bgr = [103.0626238, 115.90288257, 123.15163084]  # ResNet
        default_im_size = [224, 224]  # ResNet

        parser = argparse.ArgumentParser(description='Data Layer')
        parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=default_im_size, type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=default_mean_bgr, type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        parser.add_argument("-t", "--top_names", nargs='+', choices=top_names_choices, required=True,
                            type=str, help="ordered list of top names e.g input_image azimuth shape")
        parser.add_argument("-f", "--flip_ratio", default=0.5, type=float, help="Flip ratio in range [0, 1] (Defaults to 0.5)")
        params = parser.parse_args(param_str.split())

        print "-------------------- DataLayer Config ----------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up DataLayer ..."

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        # Store the ordered list of top_names
        self.top_names = params.top_names
        # Store batch size as member variable for use in other methods
        self.batch_size = params.batch_size
        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(params.mean_bgr).reshape(1, 3, 1, 1)
        # set network input_image size
        self.im_size = params.im_size
        # set flip_ratio
        self.flip_ratio = params.flip_ratio

        assert len(top) == len(self.top_names), 'Number of tops do not match specified top_names'

        # set top shapes
        top_shapes = {
            "input_image": (self.batch_size, 3, self.im_size[1], self.im_size[0]),
            "viewpoint": (self.batch_size, 3),
            "bbx_amodal": (self.batch_size, 4),
            "bbx_crop": (self.batch_size, 4),
            "center_proj": (self.batch_size, 2),
        }

        # Reshape tops
        for top_index, top_name in enumerate(self.top_names):
            top[top_index].reshape(*top_shapes[top_name])

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(transpose=False)

        # Create placeholder for annotations
        self.data_samples = []

        print 'DataLayer has been setup.'

    def add_dataset(self, dataset):
        """Add annotations from a json dataset"""
        print '---- Adding data from {} datatset -----'.format(dataset.name())
        print 'Number of data points (annotations) = {:,}'.format(len(self.data_samples))

        prev_num_of_images = len(self.image_loader)

        image_files = []
        for i in xrange(dataset.num_of_annotations()):
            image_info = dataset.annotations()[i]
            img_path = osp.join(dataset.rootdir(), image_info['image_file'])
            image_files.append(img_path)

            image_id = prev_num_of_images + i

            for obj_info in image_info['objects']:
                data_sample = {}
                data_sample['image_id'] = image_id
                data_sample['id'] = obj_info['id']
                data_sample['category'] = obj_info['category']
                for field in ['viewpoint', 'bbx_amodal', 'bbx_visible', 'center_proj']:
                    if field in obj_info:
                        data_sample[field] = np.array(obj_info[field])

                if 'viewpoint' in data_sample:
                    vp = data_sample['viewpoint']
                    assert (vp >= -np.pi).all() and (vp < np.pi).all(), "Bad viewpoint = {}".format(vp)

                # Add data_sample
                self.data_samples.append(data_sample)

        self.image_loader.preload_images(image_files)
        print 'Number of data points (annotations) = {:,}'.format(len(self.data_samples))
        print "--------------------------------------------------------------------"

    def generate_datum_ids(self):
        num_of_data_points = len(self.data_samples)

        # set of data indices in [0, num_of_data_points)
        self.data_ids = range(num_of_data_points)
        self.curr_data_ids_idx = 0

        # Shuffle from the begining if in the train phase
        if self.phase == caffe.TRAIN:
            shuffle(self.data_ids)

        assert len(self.data_ids) > self.batch_size, 'batch size ({})is smaller than number of data points ({}).'.format(self.batch_size, len(self.data_ids))
        print 'Total number of data points (annotations) = {:,}'.format(num_of_data_points)
        return num_of_data_points

    def augment_data_sample(self, data_idx):
        """Returns augmented data_sample and image"""
        # fetch the data sample (object)
        original_data_sample = self.data_samples[data_idx]

        full_image = self.image_loader[original_data_sample['image_id']]

        # TODO Jitter
        bbx_crop = original_data_sample['bbx_visible'].copy()

        data_sample = {}
        data_sample['id'] = original_data_sample['id']
        data_sample['category'] = original_data_sample['category']
        data_sample['bbx_crop'] = bbx_crop
        data_sample['bbx_amodal'] = original_data_sample['bbx_amodal'].copy()
        data_sample['viewpoint'] = original_data_sample['viewpoint'].copy()
        data_sample['center_proj'] = original_data_sample['center_proj'].copy()
        data_sample['input_image'] = crop_and_resize_image(full_image, bbx_crop, self.im_size)

        if random() < self.flip_ratio:
            W = full_image.shape[1]
            data_sample['bbx_crop'][[0, 2]] = W - data_sample['bbx_crop'][[2, 0]]
            data_sample['bbx_amodal'][[0, 2]] = W - data_sample['bbx_amodal'][[2, 0]]
            data_sample['center_proj'][0] = W - data_sample['center_proj'][0]
            data_sample['viewpoint'][0] = -data_sample['viewpoint'][0]
            data_sample['viewpoint'][2] = -data_sample['viewpoint'][2]
            data_sample['input_image'] = np.fliplr(data_sample['input_image'])

        # Change image channel order
        data_sample['input_image'] = data_sample['input_image'].transpose((2, 0, 1))

        return data_sample

    def forward(self, bottom, top):
        """
        Load current batch of data and labels to caffe blobs
        """

        assert hasattr(self, 'data_ids'), 'Most likely data has not been initialized before calling forward()'

        for i in xrange(self.batch_size):
            # Did we finish an epoch?
            if self.curr_data_ids_idx == len(self.data_ids):
                self.curr_data_ids_idx = 0
                shuffle(self.data_ids)

            # Current Data index
            data_idx = self.data_ids[self.curr_data_ids_idx]

            # fetch the data sample (object)
            data_sample = self.augment_data_sample(data_idx)

            # set to data
            for top_index, top_name in enumerate(self.top_names):
                top[top_index].data[i, ...] = data_sample[top_name]

            self.curr_data_ids_idx += 1

        # subtarct mean from image data blob
        if 'input_image' in self.top_names:
            top[self.top_names.index('input_image')].data[...] -= self.mean_bgr


class DataLayerAmodalCropping(DataLayer):
    """
    DataLayer which does cropping (uniform scale) via amodal bbx
    """
    def augment_data_sample(self, data_idx):
        """Returns augmented data_sample and image"""
        # fetch the data sample (object)
        original_data_sample = self.data_samples[data_idx]

        full_image = self.image_loader[original_data_sample['image_id']]

        # TODO Jitter
        bbx_crop = original_data_sample['bbx_amodal'].copy()

        data_sample = {}
        data_sample['id'] = original_data_sample['id']
        data_sample['category'] = original_data_sample['category']
        data_sample['bbx_crop'] = bbx_crop
        data_sample['bbx_amodal'] = original_data_sample['bbx_amodal'].copy()
        data_sample['viewpoint'] = original_data_sample['viewpoint'].copy()
        data_sample['center_proj'] = original_data_sample['center_proj'].copy()
        data_sample['input_image'] = uniform_crop_and_resize_image(full_image, bbx_crop, self.im_size, np.squeeze(self.mean_bgr))

        if random() < self.flip_ratio:
            W = full_image.shape[1]
            data_sample['bbx_crop'][[0, 2]] = W - data_sample['bbx_crop'][[2, 0]]
            data_sample['bbx_amodal'][[0, 2]] = W - data_sample['bbx_amodal'][[2, 0]]
            data_sample['center_proj'][0] = W - data_sample['center_proj'][0]
            data_sample['viewpoint'][0] = -data_sample['viewpoint'][0]
            data_sample['viewpoint'][2] = -data_sample['viewpoint'][2]
            data_sample['input_image'] = np.fliplr(data_sample['input_image'])

        # Change image channel order
        data_sample['input_image'] = data_sample['input_image'].transpose((2, 0, 1))

        return data_sample
