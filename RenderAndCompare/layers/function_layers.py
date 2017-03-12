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
