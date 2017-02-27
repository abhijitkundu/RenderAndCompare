import numpy as np

class BoundingBox:
    def __init__(self, min, size):
        self._min = np.array([min[0], min[1]], dtype=np.float)
        self._size = np.array([size[0], size[1]], dtype=np.float)

    @classmethod
    def fromRect(cls, rect):
        'Creates BoundingBox from a rect (x, y, w, h)'
        return cls(rect[:2], rect[2:4])

    def __repr__(self):
        'Return a nicely formatted representation string'
        return 'BoundingBox(min=%s, size=%s, center=%s)' % (self._min, self._size, self.center())

    def min(self):
        return self._min

    def max(self):
        return self._min + self._size
    
    def center(self):
        return self._min + self._size / 2

    def size(self):
        return self._size

    def translate(self, translation):
        self._min += translation



class BoundingBoxTransform:
    def __init__(self, translation, scale):
        self.translation = np.array([translation[0], translation[1]], dtype=np.float)
        self.scale = np.array([scale[0], scale[1]], dtype=np.float)
        
    @classmethod
    def fromTwoBoundingBoxes(cls, bbxA, bbxB):
        'Return aTb'
        return cls.fromNumPy((bbxB.origin - bbxA.origin)/bbxA.size, bbxB.size/bbxA.size)

    def __repr__(self):
        'Return a nicely formatted representation string'
        return 'BoundingBox(translation=%s, scale=%s)' % (self.translation, self.scale)
    
    def matrix(self):
        'Returns the transformation matrix'
        tfm = np.identity(3, dtype=np.float)
        tfm[:2, 2] = self.translation
#         tfm.diagonal()[:2] = self.scale
        tfm[0,0] = self.scale[0]
        tfm[1,1] = self.scale[1]
        return tfm

