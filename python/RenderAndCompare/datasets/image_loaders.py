"""
Image Loaders
"""

import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import tqdm


class NaiveImageLoader(object):
    """
    This is a naive imageloader
    """

    def __init__(self, transpose=True, image_files=None):
        self.transpose = transpose
        self.image_list = []
        if image_files is not None:
            self.add_images(image_files)
        print "NaiveImageLoader initialized with {:,} images".format(len(self.image_list))

    def add_images(self, image_files):
        """Adds a set of image paths to the loader"""
        print "NaiveImageLoader: Adding {:,} images".format(len(image_files))
        self.image_list.extend(image_files)
        print "NaiveImageLoader now has {:,} images".format(len(self.image_list))

    def verify_image_sizes(self, image_sizes):
        """Verify images have same size as provided by image_sizes (list of [W, H])"""
        assert len(self.image_list) == len(image_sizes)
        print "NaiveImageLoader: Checking {:,} images".format(len(self.image_list))
        parallel_read_and_verify_images(self.image_list, image_sizes, n_jobs=12)

    def __getitem__(self, index):
        return read_resize_transpose_image(self.image_list[index], [-1, -1], self.transpose)

    def __len__(self):
        return len(self.image_list)

    def __repr__(self):
        'Return a nicely formatted representation string'
        return "NaiveImageLoader(number_of_images={}, Transpose={})".format(len(self.image_list), self.transpose)


class BatchImageLoader(object):
    """
    This class first prefetches ALL the images. Can be bad for large datasets. But for datasets
    which can be fit to memory, this class ensures no disk IO except at the initialization.
    """

    def __init__(self, im_size=[-1, -1], transpose=True, image_files=None):
        assert len(im_size) == 2, 'Expects a int list/tuple of lenght 2 as im_size'
        assert all(isinstance(n, int) for n in im_size), 'Expects im_size to be in int'
        self.im_size = im_size
        self.transpose = transpose
        self.preloaded_images = []
        if image_files is not None:
            self.add_images(image_files)
        print "BatchImageLoader initialized with {:,} images".format(len(self.preloaded_images))

    def add_images(self, image_files):
        print "BatchImageLoader: Preloading {:,} images".format(len(image_files))
        self.preloaded_images += parallel_read_resize_transpose_images(image_files, self.im_size, self.transpose)
        print "BatchImageLoader now has {:,} images".format(len(self.preloaded_images))

    def crop_and_preload_images(self, image_files, cropping_boxes):
        assert len(image_files) == len(cropping_boxes), 'Number of images ({}) need to be same as numbe of boxes ({})'.format(len(image_files), len(cropping_boxes))
        print "BatchImageLoader: Crop + Preloading {:,} images".format(len(image_files))
        self.preloaded_images += parallel_read_crop_resize_transpose_images(image_files, cropping_boxes, self.im_size, self.transpose)
        print "BatchImageLoader now has {:,} images".format(len(self.preloaded_images))

    def __getitem__(self, index):
        return self.preloaded_images[index]

    def __len__(self):
        return len(self.preloaded_images)

    def __repr__(self):
        'Return a nicely formatted representation string'
        return "BatchImageLoader(number_of_images={}, Transpose={})".format(len(self.preloaded_images), self.transpose)


def makeRGB8image(image):
    rgb8_image = image.transpose(1, 2, 0)
    rgb8_image = rgb8_image[:, :, ::-1]  # change to RGB
    return np.uint8(rgb8_image)


def makeBGR8image(image):
    rgb8_image = image.transpose(1, 2, 0)
    return np.uint8(rgb8_image)


def scale_image(image, target_size, max_size):
    """uniformly scales an image's shorter size to target size bounded by max_size"""
    img_hw = image.shape[:2]
    img_size_min = np.min(img_hw)
    img_size_max = np.max(img_hw)
    img_scale = float(target_size) / float(img_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(img_scale * img_size_max) > max_size:
        img_scale = float(max_size) / float(img_size_max)
    # resize image by img_scale
    image = cv2.resize(image, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
    return image, img_scale


def crop_and_resize_image(image, cropping_box, target_size):
    assert image.ndim == 3, 'Expects image to be rank 3 tensor (color image) but got rank {}'.format(image.ndim)
    assert image.shape[2] == 3, 'Expects image to RGB image in (H, W, C) order. Current shape= {}'.format(image.shape)
    bbx = cropping_box.astype(int)
    assert (bbx[3] - bbx[1]) > 0 and (bbx[2] - bbx[0]) > 0, 'Invalid bbx = {}'.format(bbx)
    cropped_image = image[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    cropped_image = cv2.resize(cropped_image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    return cropped_image


def uniform_crop_and_resize_image(image, bbx_crop, target_size, mean_bgr):
    assert image.ndim == 3, 'Expects image to be rank 3 tensor (color image) but got rank {}'.format(image.ndim)
    assert image.shape[2] == 3, 'Expects image to RGB image in (H, W, C) order. image.shape = {}'.format(image.shape)
    assert bbx_crop.shape == (4,), 'Expects bbx to be vector of size 4. bbx_crop.shape = {}'.format(bbx_crop.shape)

    bbx_crop_wh = bbx_crop[2:] - bbx_crop[:2]
    assert np.all(bbx_crop_wh > 0), 'Invalid bbx = {}'.format(bbx_crop)

    bbx_increase = (bbx_crop_wh.max() - bbx_crop_wh) / 2

    roi_uniform = bbx_crop.copy()
    roi_uniform[:2] -= bbx_increase
    roi_uniform[2:] += bbx_increase
    roi_uniform[:2] = np.floor(roi_uniform[:2])
    roi_uniform[2:] = np.ceil(roi_uniform[2:])
    roi_uniform = roi_uniform.astype(np.int)

    canvas_wh = roi_uniform[2:] - roi_uniform[:2]
    assert np.abs(canvas_wh[0] - canvas_wh[1]) <= 1, 'canvas_wh = {}'.format(canvas_wh)

    canvas = np.empty([canvas_wh[1], canvas_wh[0], 3], dtype=np.uint8)
    canvas[..., 0] = mean_bgr[0]
    canvas[..., 1] = mean_bgr[1]
    canvas[..., 2] = mean_bgr[2]

    roi_src = roi_uniform.copy()
    roi_src[:2] = np.maximum(roi_src[:2], 0)
    roi_src[2:] = np.minimum(roi_src[2:], [[image.shape[1] - 1, image.shape[0] - 1]])

    roi_dst = roi_uniform.copy()
    roi_dst[:2] = roi_src[:2] - roi_uniform[:2]
    roi_dst[2:] = roi_dst[:2] + roi_src[2:] - roi_src[:2]

    canvas[roi_dst[1]:roi_dst[3], roi_dst[0]:roi_dst[2], :] = image[roi_src[1]:roi_src[3], roi_src[0]:roi_src[2], :]

    cropped_image = cv2.resize(canvas, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
    return cropped_image


def read_and_verify_image(image_file, image_size):
    assert osp.isfile(image_file), "Image file {} do not exist".format(image_file)
    image = cv2.imread(image_file)
    assert image.size != 0, 'Invalid image'
    expected_image_shape = (image_size[1], image_size[0], 3)
    assert image.shape == expected_image_shape, 'Loaded image shape {} does not meet expected {} for {}'.format(image.shape, expected_image_shape, image_file)


def parallel_read_and_verify_images(image_files, image_sizes, n_jobs=12):
    num_of_images = len(image_files)
    assert num_of_images == len(image_sizes)
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        for image_file, image_size in tqdm.tqdm(zip(image_files, image_sizes), total=num_of_images):
            read_and_verify_image(image_file, image_size)
        return

    #  Use with statement to ensure threads are cleaned up promptly
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # generate a list of futures by submitting jobs
        futures = [executor.submit(read_and_verify_image, image_file, image_size) for image_file, image_size in zip(image_files, image_sizes)]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for _ in tqdm.tqdm(as_completed(futures), **kwargs):
            pass
    for f in futures:
        f.result()


def read_resize_transpose_image(image_file, target_size, transpose):
    image = cv2.imread(image_file)
    assert image.size != 0, 'Invalid image'
    assert image.ndim == 3, 'Expects image to be rank 3 tensor (color image) but got rank {}'.format(image.ndim)

    if all(v > 0 for v in target_size):
        image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)

    if transpose:
        # move image channels to outermost dimension
        image = image.transpose((2, 0, 1))

    return image


def read_crop_resize_transpose_image(image_file, cropping_box, target_size, transpose):
    image = cv2.imread(image_file)
    assert image.size != 0, 'Invalid image'
    assert image.ndim == 3, 'Expects image to be rank 3 tensor (color image) but got rank {}'.format(image.ndim)

    bbx = cropping_box.astype(int)
    assert (bbx[3] - bbx[1]) > 0 and (bbx[2] - bbx[0]) > 0, 'Invalid bbx = {}'.format(bbx)
    image = image[bbx[1]:bbx[3], bbx[0]:bbx[2]]

    if all(v > 0 for v in target_size):
        image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)

    if transpose:
        # move image channels to outermost dimension
        image = image.transpose((2, 0, 1))

    return image


def parallel_read_resize_transpose_images(images, target_size, transpose, n_jobs=12, front_num=3):
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [read_resize_transpose_image(a, target_size, transpose) for a in images[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [read_resize_transpose_image(a, target_size, transpose) for a in tqdm.tqdm(images[front_num:])]
    # Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        futures = [pool.submit(read_resize_transpose_image, a, target_size, transpose) for a in images[front_num:]]
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


def parallel_read_crop_resize_transpose_images(images, cropping_boxes, target_size, transpose, n_jobs=12, front_num=3):
    num_of_images = len(images)
    assert len(cropping_boxes) == num_of_images

    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [read_crop_resize_transpose_image(images[i], cropping_boxes[i], target_size, transpose) for i in xrange(front_num)]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [read_crop_resize_transpose_image(images[i], cropping_boxes[i], target_size, transpose) for i in tqdm.trange(front_num, num_of_images)]
    # Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        futures = [pool.submit(read_crop_resize_transpose_image, images[i], cropping_boxes[i], target_size, transpose)
                   for i in xrange(front_num, num_of_images)]
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
