import os.path as osp
import argparse
from random import shuffle
import caffe
import numpy as np
from RenderAndCompare.datasets import BatchImageLoader


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
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
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
        else:
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

    def parse_param_str(self, param_str):
        top_names_choices = ['input_image', 'viewpoint', 'bbx_amodal', 'bbx_crop']
        default_mean_bgr = [103.0626238, 115.90288257, 123.15163084] # ResNet
        default_im_size = 224 # ResNet

        parser = argparse.ArgumentParser(description='Data Layer')
        parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch Size")
        parser.add_argument("-w", "--width", default=default_im_size, type=int, help="Image Width")
        parser.add_argument("-h", "--height", default=default_im_size, type=int, help="Image Height")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=default_mean_bgr, type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        parser.add_argument("-t", "--top_names", nargs='+', choices=top_names_choices, required=True, type=str, help="ordered list of top names e.g input_image azimuth shape")
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

        assert len(top) == len(self.top_names), 'Number of tops do not match specified top_names'

        # ----- Reshape tops -----#

        # Reshape image data top (B, C, H, W)
        if 'input_image' in self.top_names:
            top[self.top_names.index('input_image')].reshape(self.batch_size, 3, params.height, params.width)  # Image Data
        
        # Reshape viewpoint tops (B, 3)
        if 'viewpoint' in self.top_names:
            top[self.top_names.index('viewpoint')].reshape(self.batch_size, 3)  # viewpoint Data (a, e, t)
        
        # Reshape bbx tops (B, 4)
        if 'bbx_amodal' in self.top_names:
            top[self.top_names.index('bbx_amodal')].reshape(self.batch_size, 4)



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


class JointDataLayer(AbstractDataLayer):

    def parse_param_str(self, param_str):
        top_names_choices = ['data', 'azimuth', 'elevation', 'tilt', 'distance', 'bbx_amodal', 'bbx_crop', 'shape']
        default_mean_bgr = [103.0626238, 115.90288257, 123.15163084] # ResNet
        default_im_size = [224, 224] # ResNet

        parser = argparse.ArgumentParser(description='Joint Data Layer')

        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=default_im_size, type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=default_mean_bgr, type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        parser.add_argument("-t", "--top_names", nargs='+', choices=top_names_choices, required=True, type=str, help="ordered list of top names e.g data azimuth shape")
        
        params = parser.parse_args(param_str.split())

        print "----------------- JointDataLayer Config --------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up JointDataLayer ..."
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

        viewpoint_top_names = ['azimuth', 'elevation', 'tilt', 'distance']
        bbx_top_names = ['bbx_amodal', 'bbx_crop']

        # Reshape image data top (B, C, H, W)
        if 'data' in self.top_names:
            top[self.top_names.index('data')].reshape(self.batch_size, 3, params.im_size[1], params.im_size[0])  # Image Data

        # Reshape viewpoint tops (B, )
        for top_name in viewpoint_top_names:
            if top_name in self.top_names:
                top[self.top_names.index(top_name)].reshape(self.batch_size,)

        # Reshape bbx tops (B, 4) 
        for top_name in bbx_top_names:
            if top_name in self.top_names:
                top[self.top_names.index(top_name)].reshape(self.batch_size, 4)

        # Reshape shape top (B, 10) 
        if 'shape' in self.top_names:
                top[self.top_names.index('shape')].reshape(self.batch_size, 10)

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(params.im_size)

        # Create placeholder for GT annotations
        if any(item in self.top_names for item in viewpoint_top_names):
            self.viewpoints = []

        if 'bbx_amodal' in self.top_names:
            self.amodal_boxes = []

        if 'bbx_crop' in self.top_names:
            self.crop_boxes = []

        if 'shape' in self.top_names:
            self.shape_params = []

        print 'JointDataLayer has been setup.'

    def add_dataset(self, dataset):
        print '---- Adding data from {} datatset -----'.format(dataset.name())

        image_files = []
        for i in xrange(dataset.num_of_annotations()):
            annotation = dataset.annotations()[i]
            img_path = osp.join(dataset.rootdir(), annotation['image_file'])
            image_files.append(img_path)

            if hasattr(self, 'viewpoints'):
                viewpoint = annotation['viewpoint']
                self.viewpoints.append(viewpoint)
            
            if hasattr(self, 'amodal_boxes'):
                self.amodal_boxes.append(np.array(annotation['bbx_amodal'], dtype=np.float))
                
            if hasattr(self, 'crop_boxes'):
                self.crop_boxes.append(np.array(annotation['bbx_crop'], dtype=np.float))

            if hasattr(self, 'shape_params'):
                self.shape_params.append(np.array(annotation['shape_param'], dtype=np.float))

        self.image_loader.preload_images(image_files)
        print "--------------------------------------------------------------------"

    def generate_datum_ids(self):
        num_of_data_points = len(self.image_loader)

        for attr in ['viewpoints', 'amodal_boxes', 'crop_boxes', 'shape_params']:
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

            if hasattr(self, 'viewpoints'):
                viewpoint = self.viewpoints[data_idx]
                if 'azimuth' in self.top_names:
                    top[self.top_names.index('azimuth')].data[i, ...] = viewpoint[0]
                if 'elevation' in self.top_names:
                    top[self.top_names.index('elevation')].data[i, ...] = viewpoint[1]
                if 'tilt' in self.top_names:
                    top[self.top_names.index('tilt')].data[i, ...] = viewpoint[2]
                if 'distance' in self.top_names:
                    top[self.top_names.index('distance')].data[i, ...] = viewpoint[3]

            if hasattr(self, 'amodal_boxes'):
                top[self.top_names.index('bbx_amodal')].data[i, ...] = self.amodal_boxes[data_idx]

            if hasattr(self, 'crop_boxes'):
                top[self.top_names.index('bbx_crop')].data[i, ...] = self.crop_boxes[data_idx]

            if hasattr(self, 'shape_params'):
                top[self.top_names.index('shape')].data[i, ...] = self.shape_params[data_idx]

            self.curr_data_ids_idx += 1

        # subtarct mean from full image data blob
        if 'data' in self.top_names:          
            top[self.top_names.index('data')].data[...] -= self.mean_bgr