import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        print "BatchImageLoader: Preloading {:,} images".format(len(image_files))
        self.preloaded_images += parallel_read_resize_transpose_images(image_files, read_resize_transpose_image, self.im_size)

        print "BatchImageLoader now has {:,} images".format(len(self.preloaded_images))

    def crop_and_preload_images(self, image_files, cropping_boxes):
        assert len(image_files) == len(cropping_boxes), 'Number of images ({}) need to be same as numbe of boxes ({})'.format(len(image_files), len(cropping_boxes))
        print "BatchImageLoader: Crop + Preloading {:,} images".format(len(image_files))
        self.preloaded_images += parallel_read_crop_resize_transpose_images(image_files, cropping_boxes, self.im_size)
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


def read_resize_transpose_image(image_file, im_size):
    image = cv2.imread(image_file)
    assert image.size != 0, 'Invalid image'
    assert image.ndim == 3, 'Expects image to be rank 3 tensor (color image) but got rank {}'.format(image.ndim)

    image = cv2.resize(image, (im_size[0], im_size[1]), interpolation=cv2.INTER_LINEAR)
    # move image channels to outermost dimension
    image = image.transpose((2, 0, 1))
    return image


def read_crop_resize_transpose_image(image_file, cropping_box, im_size):
    image = cv2.imread(image_file)
    assert image.size != 0, 'Invalid image'
    assert image.ndim == 3, 'Expects image to be rank 3 tensor (color image) but got rank {}'.format(image.ndim)

    bbx = cropping_box.astype(int)
    assert (bbx[3] - bbx[1]) > 0 and (bbx[2] - bbx[0]) > 0, 'Invalid bbx = {}'.format(bbx)
    cropped_image = image[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    image = cv2.resize(cropped_image, (im_size[0], im_size[1]), interpolation=cv2.INTER_LINEAR)
    # move image channels to outermost dimension
    image = image.transpose((2, 0, 1))
    return image


def parallel_read_resize_transpose_images(images, im_size, n_jobs=12, front_num=3):
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [read_resize_transpose_image(a, im_size) for a in images[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [read_resize_transpose_image(a, im_size) for a in tqdm.tqdm(images[front_num:])]
    # Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        futures = [pool.submit(read_resize_transpose_image, a, im_size) for a in images[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm.tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm.tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def parallel_read_crop_resize_transpose_images(images, cropping_boxes, im_size, n_jobs=12, front_num=3):
    num_of_images = len(images)
    assert len(cropping_boxes) == num_of_images

    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [read_crop_resize_transpose_image(images[i], cropping_boxes[i], im_size) for i in xrange(front_num)]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [read_crop_resize_transpose_image(images[i], cropping_boxes[i], im_size) for i in tqdm.trange(front_num, num_of_images)]
    # Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        futures = [pool.submit(read_crop_resize_transpose_image, images[i], cropping_boxes[i], im_size) for i in xrange(front_num, num_of_images)]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm.tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm.tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
