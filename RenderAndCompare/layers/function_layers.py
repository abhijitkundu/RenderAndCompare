import caffe
import numpy as np
import argparse


class SoftMaxWithTemperature(caffe.Layer):

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='SoftMaxWithTemperature Layer')
        parser.add_argument("-t", "--temperature", default=1.0, type=float, help="SoftMax temperature")
        params = parser.parse_args(param_str.split())
        return params

    def setup(self, bottom, top):
        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        assert 1 <= len(bottom) <= 2, 'There should be one/two bottom blobs'
        assert len(top) == 1, 'There should be 1 top blobs'
        assert bottom[0].data.ndim == 2, 'bottom[0] layer need to be (N, K) tensor'

        if len(bottom) == 2:
            assert bottom[1].data.ndim == 1, 'bottom[0] layer need to be (N,) tensor'
            assert bottom[1].data.shape[0] == bottom[0].data.shape[0], 'shape mismatch'
        else:
            self.temperature = params.temperature

        # Top should have the same size as bottom[0]
        top[0].reshape(*bottom[0].data.shape)

        print "---------- SoftMaxWithTemperature Layer Config ---------------"
        if (len(bottom) < 2):
            print "Temperature = Fixed ({})".format(self.temperature)
        else:
            print "Temperature = bottom[1] with shape {}".format(bottom[1].data.shape)
        print "bottom[0].shape =   {}".format(bottom[0].data.shape)
        print "top[0].shape =      {}".format(top[0].data.shape)
        print "--------------------------------------------------------------"

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        max_x = np.max(bottom[0].data, axis=1).reshape((-1, 1))

        if len(bottom) == 2:
            exp_x = np.exp((bottom[0].data - max_x) / bottom[1].data.reshape((-1, 1)))
        else:
            exp_x = np.exp((bottom[0].data - max_x) / self.temperature)

        top[0].data[...] = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

    def backward(self, top, propagate_down, bottom):
        pass


class L2Normalization(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'
        assert bottom[0].data.ndim == 2

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        sq_bottom = np.square(bottom[0].data)
        self.sq_norm = np.sum(sq_bottom, axis=1, keepdims=True)
        norm = np.sqrt(self.sq_norm + np.finfo(float).eps)
        self.inv_norm = 1. / norm
        top[0].data[...] = bottom[0].data * self.inv_norm

    def backward(self, top, propagate_down, bottom):
        # diff = top[0].diff * inv_norm
        # dinv_norm = bottom[0].data
        # dnorm = -1. / (norm**2) * dinv_norm
        # dsq_norm = 0.5 * (1. / np.sqrt(sq_norm + np.finfo(float).eps)) * dnorm
        # dsq = dsq_norm
        # diff += 2 * bottom[0].data * dsq
        # bottom[0].diff[...] = diff

        diff = top[0].diff * self.inv_norm
        dnorm = -1. / self.sq_norm * bottom[0].data
        dsq_norm = 0.5 * (1. / np.sqrt(self.sq_norm + np.finfo(float).eps)) * dnorm
        diff += 2 * bottom[0].data * dsq_norm
        bottom[0].diff[...] = diff


class AngularExpectation(caffe.Layer):
    """
    Computes expected angle from prob and bin size
    Output is [cos, sin]
    """

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='AngularExpectation Layer')
        parser.add_argument("-b", "--num_of_bins", default=24, type=int, help="Number of bins")
        params = parser.parse_args(param_str.split())
        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        self.num_of_bins = params.num_of_bins
        assert bottom[0].data.ndim == 2
        assert bottom[0].data.shape[1] == self.num_of_bins
        top[0].reshape(bottom[0].data.shape[0], 2)

        angles = (2 * np.pi / self.num_of_bins) * (np.arange(0.5, self.num_of_bins))
        self.cs = np.hstack((np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)))

        print "------------- AngularExpectation Layer Config ------------------"
        print "Number of bins = {}".format(self.num_of_bins)
        print "bottom.shape =   {}".format(bottom[0].data.shape)
        print "top.shape =      {}".format(top[0].data.shape)
        print "--------------------------------------------------------------"

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # bottom_dot_cs = bottom[0].data.dot(self.cs)
        # l2_norms = np.linalg.norm(bottom_dot_cs, axis=1, keepdims=True)
        # p_dot_cs_normalized = bottom_dot_cs / l2_norms
        top[0].data[...] = bottom[0].data.dot(self.cs)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff.dot(self.cs.T)


class ConstantMultiply(caffe.Layer):
    """A layer that just multiplies by constant user specified parame"""

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='ConstantMultiply Layer')
        parser.add_argument("-m", "--multiplier", default=10, type=float, help="constant multiplication param")
        params = parser.parse_args(param_str.split())
        return params

    def setup(self, bottom, top):
        params = self.parse_param_str(self.param_str)
        self.multiplier = params.multiplier

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.multiplier * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.multiplier * top[0].diff
