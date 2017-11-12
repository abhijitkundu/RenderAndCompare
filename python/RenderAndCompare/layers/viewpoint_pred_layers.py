import numpy as np
import caffe

def softmax(x, t=1.0):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    assert x.ndim == 2
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp((x - max_x) / t)
    out = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
    return out

class AngularL1LossLayer(caffe.Layer):
    """Compute the L1 Loss for angles (in radians)"""

    def setup(self, bottom, top):
        assert len(bottom) == 2, "Need two bottoms to compute distance"
        assert len(top) == 1, "Requires a single top layer"
        assert np.squeeze(bottom[0].data).shape == np.squeeze(bottom[1].data).shape, \
            "bottom[0].shape={}, but bottom[1].shape={}".format(bottom[0].data.shape, bottom[1].data.shape)

    def reshape(self, bottom, top):
        # difference is shape of inputs
        self.diff = np.zeros_like(np.squeeze(bottom[0].data), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = (np.squeeze(bottom[0].data) - np.squeeze(bottom[1].data) + np.pi) % (2 * np.pi) - np.pi
        top[0].data[...] = np.sum(np.fabs(self.diff)) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            sign = 1 if i == 0 else -1
            alpha = sign * top[0].diff[0] / bottom[i].num
            bottom[i].diff[...] = alpha * np.sign(self.diff).reshape(bottom[i].diff.shape)


class AverageAngularError(caffe.Layer):
    """Caffe layer Computes average angular error"""

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='AverageAngularError Layer')
        parser.add_argument('--degrees_out', dest='degrees_out', action='store_true')
        parser.add_argument('--radians_out', dest='degrees_out', action='store_false')
        parser.set_defaults(degrees_out=True)
        params = parser.parse_args(param_str.split())

        return params

    def setup(self, bottom, top):
        """Setup AverageAngularError Layer"""
        assert len(bottom) == 2, 'requires bottom layers gt and pred'
        assert len(top) == 1, 'requires a single layer.top'
        params = self.parse_param_str(self.param_str)
        self.degrees_out = params.degrees_out

    def reshape(self, bottom, top):
        assert bottom[0].count == bottom[1].count, "bottom[0].shape = {}, but bottom[1].shape = {}".format(bottom[0].data.shape, bottom[1].data.shape)
        assert bottom[0].num == bottom[1].num, "bottom[0].num = {}, but bottom[1].num = {}".format(bottom[0].num, bottom[1].num)
        batch_size = bottom[0].num
        angles_per_batch = bottom[0].count / batch_size
        self.bottom_shape = (batch_size, angles_per_batch)
        top[0].reshape(1, angles_per_batch)

    def forward(self, bottom, top):
        angle_error = (bottom[1].data.reshape(*self.bottom_shape) - bottom[0].data.reshape(*self.bottom_shape) + np.pi) % (2 * np.pi) - np.pi
        if self.degrees_out:
            angle_error = np.degrees(angle_error)
        top[0].data[...] = np.mean(np.fabs(angle_error), axis=0)

    def backward(self, top, propagate_down, bottom):
        pass


class AngularExpectation(caffe.Layer):
    """
    Takes probs over set of bins and computes the complex (angular) expectation.
    Lets say we have K bins (needs be the last dimension of top blob)
    bottom[0] (N, ..., K) probs for N samples
    top[0] (N, ..., 1) gives the expected angle in degress in [-np.pi, np.pi)
    """

    def setup(self, bottom, top):
        assert len(bottom) == 1, "requires a single bottom layer"
        assert len(top) == 1, "requires a single top layer"

        num_of_bins = bottom[0].data.shape[-1]
        angles = (2 * np.pi / num_of_bins) * (np.arange(0.5, num_of_bins)) - np.pi
        assert (angles >= -np.pi).all() and (angles < np.pi).all()
        self.cs = np.concatenate((np.cos(angles)[:, np.newaxis], np.sin(angles)[:, np.newaxis]), axis=-1)

    def reshape(self, bottom, top):
        shape = list(bottom[0].data.shape)
        shape[-1] = 2
        self.prob_dot_cs = np.zeros(tuple(shape), dtype=np.float32)   # (N, ..., 2)
        shape[-1] = 1
        top[0].reshape(*shape)  # (N, ..., 1)

    def forward(self, bottom, top):
        """Take angular (complex) expectation and then return the angle"""
        # prob_dot_cs = prob * cs
        self.prob_dot_cs = bottom[0].data.dot(self.cs)
        # output = atan2(s, c)
        top[0].data[...] = np.arctan2(self.prob_dot_cs[..., 1], self.prob_dot_cs[..., 0])[..., np.newaxis]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            c2_plus_s2 = np.sum(self.prob_dot_cs**2, axis=-1)

            d_atan2 = top[0].diff * np.concatenate(((-self.prob_dot_cs[..., 1] / c2_plus_s2)[..., np.newaxis],
                                                     (self.prob_dot_cs[..., 0] / c2_plus_s2)[..., np.newaxis]),
                                                     axis = -1)
            bottom[0].diff[...] = d_atan2.dot(self.cs.T)


class QuantizeViewPoint(caffe.Layer):
    """
    Converts continious ViewPoint measurement to quantized discrete labels
    Note the original viewpoint is asumed to lie in [-pi, +pi]
    """

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='Quantize Layer')
        parser.add_argument("-b", "--num_of_bins", default=24, type=int, help="Number of bins")
        params = parser.parse_args(param_str.split())

        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)
        self.radians_per_bin = 2 * np.pi / params.num_of_bins

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.floor_divide(bottom[0].data + np.pi, self.radians_per_bin)

    def backward(self, top, propagate_down, bottom):
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

        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)
        self.radians_per_bin = 2 * np.pi / params.num_of_bins

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = (self.radians_per_bin * bottom[0].data) + self.radians_per_bin / 2.0 - np.pi

    def backward(self, top, propagate_down, bottom):
        pass





class ViewpointExpectation(caffe.Layer):
    """
    Takes probs over set of bins and computes the complex (angular) expectation
    of a viewpoint angle. Combines AngularExpectation and L2Normalization layer
    Lets say we have K bins
    bottom[0] (N,K) probs for N samples
    top[0] (N,2) where each row is [cos, sin] of the expected angle
    top[1] (N,) is optional gives the angle in degress in [0, 360)
    """

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='ViewpointExpectation Layer')
        parser.add_argument("-b", "--num_of_bins", default=24, type=int, help="Number of bins")
        params = parser.parse_args(param_str.split())
        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single bottom blobs'
        assert 1 <= len(top) <= 2, 'requires a 1/2 top blobs'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)
        self.num_of_bins = params.num_of_bins

        assert bottom[0].data.ndim == 2
        assert bottom[0].data.shape[1] == self.num_of_bins
        top[0].reshape(bottom[0].data.shape[0], 2)

        # We have top[1]
        if len(top) == 2:
            top[1].reshape(bottom[0].data.shape[0],)

        angles = (2 * np.pi / self.num_of_bins) * (np.arange(0.5, self.num_of_bins))
        self.cs = np.hstack((np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)))

        print "----------- ViewpointExpectation Layer Config ----------------"
        print "Number of bins = {}".format(self.num_of_bins)
        print "bottom.shape =   {}".format(bottom[0].data.shape)
        print "top[0].shape =   {}".format(top[0].data.shape)
        if len(top) == 2:
            print "top[1].shape =   {}".format(top[1].data.shape)
        print "--------------------------------------------------------------"

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        self.dot = bottom[0].data.copy().dot(self.cs)
        self.sq_norm = np.sum(np.square(self.dot), axis=1, keepdims=True)
        self.inv_norm = 1. / np.sqrt(self.sq_norm)
        top[0].data[...] = self.dot * self.inv_norm
        if len(top) == 2:
            top[1].data[...] = (np.arctan2(self.dot[:, 1], self.dot[:, 0]) * 180 / np.pi) % 360.0

    def backward(self, top, propagate_down, bottom):
        """
        Lame implementation by loop over samples
        """
        top_diff = top[0].diff.copy()
        num_samples = bottom[0].data.shape[0]
        bottom_diff = np.zeros_like(bottom[0].diff, dtype=np.float32)

        # (z_1^2 + z_2^2) ^ {3/2} \in (N, 1)
        z1, z2 = self.dot[:, 0], self.dot[:, 1]
        c1, c2 = self.cs[:, 0], self.cs[:, 1]
        diff1, diff2 = top_diff[:, 0], top_diff[:, 1]
        z_factor = self.inv_norm**3

        # loop over all samples individually
        for idx in xrange(num_samples):
            d1 = (c1 * (z2[idx]**2) - c2 * z1[idx] * z2[idx]) * z_factor[idx]
            d2 = (c2 * (z1[idx]**2) - c1 * z1[idx] * z2[idx]) * z_factor[idx]
            bottom_diff[idx, :] = diff1[idx] * d1 + diff2[idx] * d2

        bottom[0].diff[...] = bottom_diff


class SoftMaxViewPoint(caffe.Layer):
    """
    Converts unormalized log probs of viewpoint estimate to a continious ViewPoint estimate
    controlled by softmax temeperature
    Has two params num_of_bins and softmax_temperature t
    """

    def parse_param_str(self, param_str):
        parser = argparse.ArgumentParser(description='SoftMaxViewPoint Layer')
        parser.add_argument("-b", "--num_of_bins", default=24, type=int, help="Number of bins")
        parser.add_argument("-t", "--temperature", default=1.0, type=float, help="SoftMax temperature")
        params = parser.parse_args(param_str.split())
        return params

    def setup(self, bottom, top):
        assert len(bottom) == 1, 'requires a single layer.bottom'
        assert len(top) == 1, 'requires a single layer.top'

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        self.num_of_bins = params.num_of_bins
        self.temperature = params.temperature

        assert self.temperature >= 0
        assert bottom[0].data.ndim == 2
        assert bottom[0].data.shape[1] == self.num_of_bins
        top[0].reshape(bottom[0].data.shape[0],)

        angles = (2 * np.pi / self.num_of_bins) * (np.arange(0.5, self.num_of_bins))
        self.centers = np.exp(1j * angles)

        print "------------- SoftMaxViewPoint Layer Config ------------------"
        print "Number of bins = {}".format(self.num_of_bins)
        print "Temperature =    {}".format(self.temperature)
        print "bottom.shape =   {}".format(bottom[0].data.shape)
        print "top.shape =      {}".format(top[0].data.shape)
        print "--------------------------------------------------------------"

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # First do softmax with temperature t
        prob = softmax(bottom[0].data, self.temperature)
        # Take complex expectation and then return the angle in degrees
        top[0].data[...] = np.angle(prob.dot(self.centers), deg=True) % 360.0

    def backward(self, top, propagate_down, bottom):
        pass
