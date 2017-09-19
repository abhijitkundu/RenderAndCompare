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


class BoxOverlapIoU(caffe.Layer):
    """
    Computes IoU overlap score between two set of boxes
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers bbx_gt and box_pred'
        assert len(top) == 1, 'requires a single top layer'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0]
        assert bottom[0].data.shape[1] == 4
        assert bottom[1].data.shape[1] == 4
        top[0].reshape(bottom[0].data.shape[0],)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x1 = np.maximum(bottom[0].data[:, 0], bottom[1].data[:, 0])
        y1 = np.maximum(bottom[0].data[:, 1], bottom[1].data[:, 1])

        x2 = np.minimum(bottom[0].data[:, 2], bottom[1].data[:, 2])
        y2 = np.minimum(bottom[0].data[:, 3], bottom[1].data[:, 3])

        # compute width and height of overlapping area
        w = x2 - x1
        h = y2 - y1

        # get overlapping areas
        inter = w * h
        a_area = (bottom[0].data[:, 2:] -  bottom[0].data[:, :2]).prod(axis=1)
        b_area = (bottom[1].data[:, 2:] -  bottom[1].data[:, :2]).prod(axis=1)
        ious = inter / (a_area + b_area - inter)

        # set invalid entries to 0 overlap
        ious[w <= 0] = 0
        ious[h <= 0] = 0

        top[0].data[...] = ious

    def backward(self, top, propagate_down, bottom):
        pass


class AverageBoxOverlapIoU(caffe.Layer):
    """
    Computes average IoU overlap score across batch of (box pairs)
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers bbx_gt and box_pred'
        assert len(top) == 1, 'requires a single top layer'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0]
        assert bottom[0].data.shape[1] == 4
        assert bottom[1].data.shape[1] == 4
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x1 = np.maximum(bottom[0].data[:, 0], bottom[1].data[:, 0])
        y1 = np.maximum(bottom[0].data[:, 1], bottom[1].data[:, 1])

        x2 = np.minimum(bottom[0].data[:, 2], bottom[1].data[:, 2])
        y2 = np.minimum(bottom[0].data[:, 3], bottom[1].data[:, 3])

        # compute width and height of overlapping area
        w = x2 - x1
        h = y2 - y1

        # get overlapping areas
        inter = w * h
        a_area = (bottom[0].data[:, 2:] -  bottom[0].data[:, :2]).prod(axis=1)
        b_area = (bottom[1].data[:, 2:] -  bottom[1].data[:, :2]).prod(axis=1)
        ious = inter / (a_area + b_area - inter)

        # set invalid entries to 0 overlap
        ious[w <= 0] = 0
        ious[h <= 0] = 0

        top[0].data[...] = np.mean(ious)

    def backward(self, top, propagate_down, bottom):
        pass