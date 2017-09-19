import numpy as np
import caffe

class GenerateCropTransformationTargets(caffe.Layer):
    """
    Computes the relative bbx transfromation aTc (log adjsuted) from two bounding boxes bbx_a and bbx_c
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers box_a and box_c'
        assert len(top) == 1, 'requires a single layer.top'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0]
        assert bottom[0].data.shape[1] == 4
        assert bottom[1].data.shape[1] == 4
        top[0].reshape(bottom[0].data.shape[0], 4)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[:, :2] = (bottom[1].data[:, :2] - bottom[0].data[:, :2]) / bottom[0].data[:, 2:]
        top[0].data[:, 2:] = np.log(bottom[1].data[:, 2:] / bottom[0].data[:, 2:])

    def backward(self, top, propagate_down, bottom):
        pass


class InvertCropTransformationTargets(caffe.Layer):
    """
    Computes the bbx_a from aTc (log adjsuted) and bbx_c
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers aTc and box_c'
        assert len(top) == 1, 'requires a single layer.top'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0]
        assert bottom[0].data.shape[1] == 4
        assert bottom[1].data.shape[1] == 4
        top[0].reshape(bottom[0].data.shape[0], 4)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[:, 2:] = bottom[1].data[:, 2:] / np.exp(bottom[0].data[:, 2:])
        top[0].data[:, :2] = bottom[1].data[:, :2] - top[0].data[:, 2:] * bottom[0].data[:, :2]

    def backward(self, top, propagate_down, bottom):
        pass


class GenerateBoxOffsetTargets(caffe.Layer):
    """
    Computes the relative bbx transfromation aTc (position_offset and scale_offset) from two bounding boxes bbx_a and bbx_c
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two bottom layers box_a and box_c'
        assert len(top) == 2, 'requires two top layers position_offset and scale_offset'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0]
        assert bottom[0].data.shape[1] == 4
        assert bottom[1].data.shape[1] == 4
        top[0].reshape(bottom[0].data.shape[0], 2)
        top[1].reshape(bottom[0].data.shape[0], 2)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[...] = (bottom[1].data[:, :2] - bottom[0].data[:, :2]) / bottom[0].data[:, 2:]
        top[1].data[...] = bottom[1].data[:, 2:] / bottom[0].data[:, 2:]

    def backward(self, top, propagate_down, bottom):
        pass


class InvertBoxOffsetTargets(caffe.Layer):
    """
    Computes the bbx_a from aTc (position_offset and scale_offset) and bbx_c
    """

    def setup(self, bottom, top):
        assert len(bottom) == 3, 'requires three bottom layers position_offset, scale_offset, and box_c'
        assert len(top) == 1, 'requires a single top layer box_a'
        assert bottom[0].data.ndim == 2
        assert bottom[1].data.ndim == 2
        assert bottom[2].data.ndim == 2
        assert bottom[0].data.shape[0] == bottom[1].data.shape[0] == bottom[2].data.shape[0]
        assert bottom[0].data.shape[1] == bottom[1].data.shape[1] == 2
        assert bottom[2].data.shape[1] == 4
        top[0].reshape(bottom[0].data.shape[0], 4)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        """
        s_a = s_c / s
        p_a = p_c - s_a * p
        """
        top[0].data[:, 2:] = bottom[2].data[:, 2:] / bottom[1].data
        top[0].data[:, :2] = bottom[2].data[:, :2] - top[0].data[:, 2:] * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        pass


class BoxOverlapIoU(caffe.Layer):
    """
    Computes IoU overlap score between two boxes
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

        x2 = np.minimum(bottom[0].data[:, 0] + bottom[0].data[:, 2], bottom[1].data[:, 0] + bottom[1].data[:, 2])
        y2 = np.minimum(bottom[0].data[:, 1] + bottom[0].data[:, 3], bottom[1].data[:, 1] + bottom[1].data[:, 3])

        # compute width and height of overlapping area
        w = x2 - x1
        h = y2 - y1

        # get overlapping areas
        inter = w * h
        a_area = bottom[0].data[:, 2] * bottom[0].data[:, 3]
        b_area = bottom[1].data[:, 2] * bottom[1].data[:, 3]
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

        x2 = np.minimum(bottom[0].data[:, 0] + bottom[0].data[:, 2], bottom[1].data[:, 0] + bottom[1].data[:, 2])
        y2 = np.minimum(bottom[0].data[:, 1] + bottom[0].data[:, 3], bottom[1].data[:, 1] + bottom[1].data[:, 3])

        # compute width and height of overlapping area
        w = x2 - x1
        h = y2 - y1

        # get overlapping areas
        inter = w * h
        a_area = bottom[0].data[:, 2] * bottom[0].data[:, 3]
        b_area = bottom[1].data[:, 2] * bottom[1].data[:, 3]
        ious = inter / (a_area + b_area - inter)

        # set invalid entries to 0 overlap
        ious[w <= 0] = 0
        ious[h <= 0] = 0

        top[0].data[...] = np.mean(ious)

    def backward(self, top, propagate_down, bottom):
        pass
