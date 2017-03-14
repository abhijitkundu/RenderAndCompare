import caffe
import numpy as np
import argparse


class DataLayer(caffe.Layer):
    """
    Generic Data layer for caffe
    """

    def parse_param_str(self, param_str):
        """
        parse the param string passed from prtotxt file. Reimplement this if rerquired.
        """
        parser = argparse.ArgumentParser(description='View Prediction Data Layer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=[227, 227], type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[104.00698793, 116.66876762, 122.67891434], type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- DataLayer Config ------------------"
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

        print "DataLayer has been setup."

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
