""" 
Coordinate transformation layers
"""

import argparse
import numpy as np
import caffe


class Coord2DTransformation(caffe.Layer):
    def parse_param_str(self, param_str, num_of_coords):
        parser = argparse.ArgumentParser(description='Coord2DTransformation Layer')
        parser.add_argument("-o", "--offsets", nargs=num_of_coords, required=True, type=float, help="offsets for each coord in bottom[0]")
        params = parser.parse_args(param_str.split())
        self.offsets = np.array(params.offsets).reshape(1, -1, 1)
        
        
    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers coords and box_c'
        assert len(top) == 1, 'requires a single top layer'

        assert bottom[0].data.ndim == bottom[1].data.ndim == 2, "bottoms need 2o be 2D tensors (N, 2) or (N, 4)"
        assert bottom[0].num == bottom[0].num, "bottom batch sizes need to be same"
        assert bottom[0].data.shape[1] % 2 == 0, "bottom[0] is expected to carry 2D coords (so multiple of 2)"
        assert bottom[1].data.shape[1] == 4, "bottom[1] is expected to carry bbx coords (so 4)"

        num_of_coords = bottom[0].shape[1] / 2
        self.parse_param_str(self.param_str, num_of_coords)
        assert self.offsets.shape == (1, num_of_coords, 1)
    
    def reshape(self, bottom, top):
        # top shape is same as bottom[0] shape
        top[0].reshape(*bottom[0].data.shape)
    
    def forward(self, bottom, top):
        B =  bottom[1].data.shape[0]    # batch_size
        C = self.offsets.shape[1]       # num_of_coords
        WH = (bottom[1].data[:, 2:] - bottom[1].data[:, :2]).reshape(-1, 1, 2)
        top[0].data[...] = ((bottom[0].data.reshape(B, C, 2) - bottom[1].data[:, :2].reshape(B, 1, 2)) / WH - self.offsets).reshape(B, 2*C)
    
    def backward(self, top, propagate_down, bottom):
        """No backward pass"""
        pass


class Coord2DTransformationInverse(caffe.Layer):
    def parse_param_str(self, param_str, num_of_coords):
        parser = argparse.ArgumentParser(description='Coord2DTransformationInverse Layer')
        parser.add_argument("-o", "--offsets", nargs=num_of_coords, required=True, type=float, help="offsets for each coord in bottom[0]")
        params = parser.parse_args(param_str.split())
        self.offsets = np.array(params.offsets).reshape(1, -1, 1)
        
        
    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers coords and box_c'
        assert len(top) == 1, 'requires a single top layer'

        assert bottom[0].data.ndim == bottom[1].data.ndim == 2, "bottoms need 2o be 2D tensors (N, 2) or (N, 4)"
        assert bottom[0].num == bottom[0].num, "bottom batch sizes need to be same"
        assert bottom[0].data.shape[1] % 2 == 0, "bottom[0] is expected to carry 2D coords (so multiple of 2)"
        assert bottom[1].data.shape[1] == 4, "bottom[1] is expected to carry bbx coords (so 4)"

        num_of_coords = bottom[0].shape[1] / 2
        self.parse_param_str(self.param_str, num_of_coords)
        assert self.offsets.shape == (1, num_of_coords, 1)
    
    def reshape(self, bottom, top):
        # top shape is same as bottom[0] shape
        top[0].reshape(*bottom[0].data.shape)
    
    def forward(self, bottom, top):
        B =  bottom[1].data.shape[0]    # batch_size
        C = self.offsets.shape[1]       # num_of_coords
        WH = (bottom[1].data[:, 2:] - bottom[1].data[:, :2]).reshape(-1, 1, 2)
        top[0].data[...] = ((bottom[0].data.reshape(B, C, 2) + self.offsets) * WH + bottom[1].data[:, :2].reshape(B, 1, 2)).reshape(B, 2*C)
    
    def backward(self, top, propagate_down, bottom):
        """No backward pass"""
        pass