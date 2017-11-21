""" 
BoundingBox helper
"""

import numpy as np


def bbx_iou_overlap(bbxA, bbxB):
    """returns a iou overlap of two bounding boxes"""
    assert bbxA.shape == bbxB.shape
    ndim = bbxA.size / 2
    x1y1 = np.maximum(bbxA[:ndim], bbxB[:ndim])
    x2y2 = np.minimum(bbxA[ndim:], bbxB[ndim:])

    wh = x2y2 - x1y1
    if np.any(wh < 0):
        return 0

    inter = wh.prod()
    a_area = (np.asarray(bbxA[ndim:]) - np.asarray(bbxA[:ndim])).prod()
    b_area = (np.asarray(bbxB[ndim:]) - np.asarray(bbxB[:ndim])).prod()
    iou = inter / (a_area + b_area - inter)
    return iou


def create_jittered_bbx(bbx, min_jittter_iou, num_bins=20):
    """returns a jittered bbx which has overlap > min_jittter_iou with input bbx"""
    assert 0 < min_jittter_iou <= 1.0
    assert bbx.size % 2 == 0
    ndim = bbx.size / 2

    bin_width = (1. - min_jittter_iou) / num_bins

    target_bin = np.random.randint(num_bins)
    target_iou_min = min_jittter_iou + target_bin * bin_width
    target_iou_max = target_iou_min + bin_width

    assert target_iou_min >= min_jittter_iou
    assert target_iou_max <= 1.

    wh = (np.asarray(bbx[ndim:]) - np.asarray(bbx[:ndim])) * (1.0 - target_iou_min)
    assert np.all(wh >= 0)
    while True:
        jitter = (np.random.normal(0.0, 0.4, (2, ndim)) * wh).reshape(2 * ndim, )
        jittered_bbx = bbx + jitter
        jittered_bbx_wh = jittered_bbx[ndim:] - jittered_bbx[:ndim]
        if np.any(jittered_bbx_wh <= 0):
            continue

        actual_iou = bbx_iou_overlap(bbx, jittered_bbx)
        if target_iou_min <= actual_iou <= target_iou_max:
            return jittered_bbx


class BoundingBox(object):
    """Aligned BoundingBox class"""

    def __init__(self, bbx_min, bbx_max):
        self._min = np.array(bbx_min, dtype=np.float)
        self._max = np.array(bbx_max, dtype=np.float)
        assert np.all(self._min <= self._max)

    @classmethod
    def fromRect(cls, rect):
        'Creates BoundingBox from a rect (x, y, w, h)'
        assert rect.size % 2 == 0
        ndim = rect.size / 2
        return cls(rect[:ndim], np.asarray(rect[:ndim]) + np.asarray(rect[ndim:]))

    @classmethod
    def fromArray(cls, bbx):
        'Creates BoundingBox from a rect (x, y, w, h)'
        assert bbx.size % 2 == 0
        ndim = bbx.size / 2
        return cls(bbx[:ndim], bbx[ndim:])

    def __repr__(self):
        'Return a nicely formatted representation string'
        return "BoundingBox(min={}, max={}, size={})".format(self._min, self._max, self.size())

    def min(self):
        """returns min extent"""
        return self._min

    def max(self):
        """returns max extent"""
        return self._max

    def center(self):
        """returns bbx center"""
        return (self._min + self._max) / 2

    def size(self):
        """returns bbx size"""
        return self._max - self._min

    def array(self):
        """Returns bbx as single array [x1, y1, x2, y2] """
        return np.concatenate((self._min, self._max))

    def translate(self, translation):
        """Translate the bounding box"""
        self._min += translation
        self._max += translation


def clip_bbx_by_image_size(bbx, width, height):
    """Clip a bbx to lie entirely inside image"""
    clipped_bbx = np.empty_like(bbx)
    clipped_bbx[:2] = np.maximum(bbx[:2], [0.0, 0.0])
    clipped_bbx[2:] = np.minimum(bbx[2:], [width, height])
    return clipped_bbx
