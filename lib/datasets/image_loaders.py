import numpy as np
import cv2

class LazyImageLoader(object):
    """
    This class does the loading of images lazily when the file is requested.
    Does not hold any image data.
    """
    def __init__(self, image_files, im_size):
        self.image_files = image_files
        self.im_size = im_size
        print "LazyImageLoader initialized with {} images".format(len(self.image_files))

    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index])
        image = cv2.resize(image, (self.im_size[0], self.im_size[1]), interpolation=cv2.INTER_LINEAR)
        image = image.transpose((2, 0, 1)) # move image channels to outermost dimension
        return image.astype(np.float32)

    def __len__(self):
        return len(self.image_files)


class BatchImageLoader(object):
    """
    This class first prefetches ALL the images. Can be bad for large datasets. But for datasets
    which can be fit to memory, this class ensures no disk IO except at the initialization.
    """
    def __init__(self, image_files, im_size):
        print "BatchImageLoader: Preloading {} images".format(len(image_files))

        # TODO Fill the preloaded_images in parallel
        self.preloaded_images = []
        for i in xrange(len(image_files)):
            image = cv2.imread(image_files[i])
            image = cv2.resize(image, (im_size[0], im_size[1]), interpolation=cv2.INTER_LINEAR)
            image = image.transpose((2, 0, 1)) # move image channels to outermost dimension
            self.preloaded_images.append(image)

        print "BatchImageLoader initialized with {} images".format(len(self.preloaded_images))

    def __getitem__(self, index):
        return self.preloaded_images[index].astype(np.float32)

    def __len__(self):
        return len(self.preloaded_images)


def makeRGB8image(image):
    rgb8_image = image.transpose(1, 2, 0)
    rgb8_image = rgb8_image[:, :, ::-1] # change to RGB
    return np.uint8(rgb8_image)

def makeBGR8image(image):
    rgb8_image = image.transpose(1, 2, 0)
    return np.uint8(rgb8_image)