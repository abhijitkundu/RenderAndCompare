import numpy as np
import cv2
import tqdm


class LazyImageLoader(object):
    """
    This class does the loading of images lazily when the file is requested.
    Does not hold any image data.
    """

    def __init__(self, im_size, image_files):
        assert len(im_size) == 2, 'Expects a int list/tuple of lenght 2 as im_size'
        assert all(isinstance(n, int) for n in im_size), 'Expects im_size to be in int'
        self.im_size = im_size
        self.image_files = image_files
        print "LazyImageLoader initialized with {} images".format(len(self.image_files))

    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index])
        image = cv2.resize(image, (self.im_size[0], self.im_size[1]), interpolation=cv2.INTER_LINEAR)
        # move image channels to outermost dimension
        image = image.transpose((2, 0, 1))
        return image.astype(np.float32)

    def __len__(self):
        return len(self.image_files)


class BatchImageLoader(object):
    """
    This class first prefetches ALL the images. Can be bad for large datasets. But for datasets
    which can be fit to memory, this class ensures no disk IO except at the initialization.
    """

    def __init__(self, im_size, image_files=None):
        assert len(im_size) == 2, 'Expects a int list/tuple of lenght 2 as im_size'
        assert all(isinstance(n, int) for n in im_size), 'Expects im_size to be in int'
        self.im_size = im_size
        self.preloaded_images = []
        if image_files is not None:
            self.preload_images(image_files)
        print "BatchImageLoader initialized with {:,} images".format(len(self.preloaded_images))

    def preload_images(self, image_files):
        # TODO Fill the preloaded_images in parallel
        print "BatchImageLoader: Preloading {:,} images".format(len(image_files))
        for i in tqdm.trange(len(image_files)):
            image = cv2.imread(image_files[i])
            image = cv2.resize(image, (self.im_size[0], self.im_size[1]), interpolation=cv2.INTER_LINEAR)
            # move image channels to outermost dimension
            image = image.transpose((2, 0, 1))
            self.preloaded_images.append(image)
        print "BatchImageLoader now has {:,} images".format(len(self.preloaded_images))

    def __getitem__(self, index):
        return self.preloaded_images[index].astype(np.float32)

    def __len__(self):
        return len(self.preloaded_images)


def makeRGB8image(image):
    rgb8_image = image.transpose(1, 2, 0)
    rgb8_image = rgb8_image[:, :, ::-1]  # change to RGB
    return np.uint8(rgb8_image)


def makeBGR8image(image):
    rgb8_image = image.transpose(1, 2, 0)
    return np.uint8(rgb8_image)
