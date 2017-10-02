import argparse
import os.path as osp
from random import shuffle, random, randint

import numpy as np

import caffe
from RenderAndCompare.datasets import (
    BatchImageLoader,
    crop_and_resize_image,
    uniform_crop_and_resize_image,
    scale_image, sample_object_infos
)

class AbstractDataLayer(caffe.Layer):
    """
    Generic python Data layer for caffe
    """

    def parse_param_str(self, param_str):
        """
        parse the param string passed from prtotxt file. Reimplement this if rerquired.
        """
        parser = argparse.ArgumentParser(description='AbstractDataLayer')
        parser.add_argument("-b", "--batch_size", default=50, type=int, help="Batch Size")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=[103.0626238, 115.90288257, 123.15163084],
                            type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        params = parser.parse_args(param_str.split())

        print "------------- AbstractDataLayer Config ------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        """Reimplement Layer setup in this method"""
        # params is expected as argparse style string
        self.params = self.parse_param_str(self.param_str)

        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(self.params.mean_bgr).reshape(1, 3, 1, 1)

        print "AbstractDataLayer has been setup."
        pass

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def forward(self, bottom, top):
        """
        Reimplement this
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

    def generate_datum_ids(self):
        """Reimplement this"""
        pass

    def number_of_datapoints(self):
        """Number of data points (samples)"""
        if hasattr(self, 'data_ids'):
            return len(self.data_ids)
        return 0

    def make_rgb8_from_blob(self, blob_data):
        """
        conveneince method to get an rgb8 image (compatible for Matplotlib display)
        """
        assert blob_data.ndim == 3, 'expects a color image (dim: 3)'
        image = blob_data + self.mean_bgr.reshape(3, 1, 1)
        image = image.transpose(1, 2, 0)
        image = image[:, :, ::-1]  # change to RGB
        return np.uint8(image)

    def make_bgr8_from_blob(self, blob_data):
        """
        conveneince method to get an bgr8 image (compatible for OpenCV display)
        """
        assert blob_data.ndim == 3, 'expects a color image (dim: 3)'
        image = blob_data + self.mean_bgr.reshape(3, 1, 1)
        image = image.transpose(1, 2, 0)
        return np.uint8(image)


class RCNNDataLayer(AbstractDataLayer):
    """Data Layer RCNN style. Inherits AbstractDataLayer"""

    def parse_param_str(self, param_str):
        top_names_choices = ['input_image', 'viewpoint', 'bbx_amodal', 'bbx_crop', 'center_proj']
        default_mean_bgr = [103.0626238, 115.90288257, 123.15163084]  # ResNet
        default_im_size = [224, 224]  # ResNet

        parser = argparse.ArgumentParser(description='RCNN style Data Layer')
        parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch Size")
        parser.add_argument("-wh", "--im_size", nargs=2, default=default_im_size, type=int, metavar=('WIDTH', 'HEIGHT'), help="Image Size [width, height]")
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=default_mean_bgr, type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        parser.add_argument("-t", "--top_names", nargs='+', choices=top_names_choices, required=True,
                            type=str, help="ordered list of top names e.g input_image azimuth shape")
        parser.add_argument("-f", "--flip_ratio", default=0.5, type=float, help="Flip ratio in range [0, 1] (Defaults to 0.5)")
        parser.add_argument("-c", "--crop_target", default='bbx_visible',
                            choices=['bbx_amodal', 'bbx_visible'], type=str, help="bbx type used for cropping (Defaults to bbx_visible)")
        parser.add_argument("-u", "--uniform_crop", action='store_true', default=False, help="If set we do not change the aspect ratio while cropping")
        params = parser.parse_args(param_str.split())

        print "-------------------- RCNNDataLayer Config ----------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up RCNNDataLayer ..."

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        # Store the ordered list of top_names
        self.top_names = params.top_names
        # Store batch size as member variable for use in other methods
        self.batch_size = params.batch_size
        # create mean bgr to directly operate on image data blob
        self.mean_bgr = np.array(params.mean_bgr).reshape(1, 3, 1, 1)
        # set network input_image size
        self.im_size = params.im_size
        # set flip_ratio
        self.flip_ratio = params.flip_ratio
        # set crop_target
        self.crop_target = params.crop_target
        # set uniform_crop
        self.uniform_crop = params.uniform_crop

        assert len(top) == len(self.top_names), 'Number of tops do not match specified top_names'

        # set top shapes
        top_shapes = {
            "input_image": (self.batch_size, 3, self.im_size[1], self.im_size[0]),
            "viewpoint": (self.batch_size, 3),
            "bbx_amodal": (self.batch_size, 4),
            "bbx_crop": (self.batch_size, 4),
            "center_proj": (self.batch_size, 2),
        }

        # Reshape tops
        for top_index, top_name in enumerate(self.top_names):
            top[top_index].reshape(*top_shapes[top_name])

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(transpose=False)

        # Create placeholder for annotations
        self.data_samples = []

        print 'RCNNDataLayer has been setup.'

    def add_dataset(self, dataset):
        """Add annotations from a json dataset"""
        print '---- Adding data from {} datatset -----'.format(dataset.name())
        print 'Number of data points (annotations) = {:,}'.format(len(self.data_samples))

        prev_num_of_images = len(self.image_loader)

        image_files = []
        for i in xrange(dataset.num_of_annotations()):
            image_info = dataset.annotations()[i]
            img_path = osp.join(dataset.rootdir(), image_info['image_file'])
            image_files.append(img_path)

            image_id = prev_num_of_images + i

            for obj_info in image_info['objects']:
                data_sample = {}
                data_sample['image_id'] = image_id
                data_sample['id'] = obj_info['id']
                data_sample['category'] = obj_info['category']
                for field in ['viewpoint', 'bbx_amodal', 'bbx_visible', 'center_proj']:
                    if field in obj_info:
                        data_sample[field] = np.array(obj_info[field])

                if 'viewpoint' in data_sample:
                    vp = data_sample['viewpoint']
                    assert (vp >= -np.pi).all() and (vp < np.pi).all(), "Bad viewpoint = {}".format(vp)

                # Add data_sample
                self.data_samples.append(data_sample)

        self.image_loader.preload_images(image_files)
        print 'Number of data points (annotations) = {:,}'.format(len(self.data_samples))
        print "--------------------------------------------------------------------"

    def generate_datum_ids(self):
        num_of_data_points = len(self.data_samples)

        # set of data indices in [0, num_of_data_points)
        self.data_ids = range(num_of_data_points)
        self.curr_data_ids_idx = 0

        # Shuffle from the begining if in the train phase
        if self.phase == caffe.TRAIN:
            shuffle(self.data_ids)

        assert len(self.data_ids) > self.batch_size, 'batch size ({})is smaller than number of data points ({}).'.format(self.batch_size, len(self.data_ids))
        print 'Total number of data points (annotations) = {:,}'.format(num_of_data_points)
        return num_of_data_points

    def augment_data_sample(self, data_idx):
        """Returns augmented data_sample and image"""
        # fetch the data sample (object)
        original_data_sample = self.data_samples[data_idx]

        full_image = self.image_loader[original_data_sample['image_id']]

        # TODO Jitter
        bbx_crop = original_data_sample[self.crop_target].copy()

        data_sample = {}
        data_sample['id'] = original_data_sample['id']
        data_sample['category'] = original_data_sample['category']
        data_sample['bbx_crop'] = bbx_crop
        data_sample['bbx_amodal'] = original_data_sample['bbx_amodal'].copy()
        data_sample['viewpoint'] = original_data_sample['viewpoint'].copy()
        data_sample['center_proj'] = original_data_sample['center_proj'].copy()
        if self.uniform_crop:
            data_sample['input_image'] = uniform_crop_and_resize_image(full_image, bbx_crop, self.im_size, np.squeeze(self.mean_bgr))
        else:
            data_sample['input_image'] = crop_and_resize_image(full_image, bbx_crop, self.im_size)

        if random() < self.flip_ratio:
            W = full_image.shape[1]
            data_sample['bbx_crop'][[0, 2]] = W - data_sample['bbx_crop'][[2, 0]]
            data_sample['bbx_amodal'][[0, 2]] = W - data_sample['bbx_amodal'][[2, 0]]
            data_sample['center_proj'][0] = W - data_sample['center_proj'][0]
            data_sample['viewpoint'][[0, 2]] = -data_sample['viewpoint'][[0, 2]]
            data_sample['input_image'] = np.fliplr(data_sample['input_image'])

        # Change image channel order
        data_sample['input_image'] = data_sample['input_image'].transpose((2, 0, 1))

        return data_sample

    def forward(self, bottom, top):
        """
        Load current batch of data and labels to caffe blobs
        """

        assert hasattr(self, 'data_ids'), 'Most likely data has not been initialized before calling forward()'

        for i in xrange(self.batch_size):
            # Did we finish an epoch?
            if self.curr_data_ids_idx == len(self.data_ids):
                self.curr_data_ids_idx = 0
                shuffle(self.data_ids)

            # Current Data index
            data_idx = self.data_ids[self.curr_data_ids_idx]

            # fetch the data sample (object)
            data_sample = self.augment_data_sample(data_idx)

            # set to data
            for top_index, top_name in enumerate(self.top_names):
                top[top_index].data[i, ...] = data_sample[top_name]

            self.curr_data_ids_idx += 1

        # subtarct mean from image data blob
        if 'input_image' in self.top_names:
            top[self.top_names.index('input_image')].data[...] -= self.mean_bgr


class FastRCNNDataLayer(AbstractDataLayer):
    """Data Layer Fast-RCNN style. Inherits AbstractDataLayer"""

    def parse_param_str(self, param_str):
        """Parse params"""
        top_names_choices = ['input_image', 'roi', 'viewpoint', 'bbx_amodal', 'bbx_crop', 'center_proj']
        default_mean_bgr = [103.0626238, 115.90288257, 123.15163084]  # ResNet
        default_size_range = [600, 1024]

        parser = argparse.ArgumentParser(description='Fast RCNN style Data Layer')
        parser.add_argument("-m", "--mean_bgr", nargs=3, default=default_mean_bgr, type=float, metavar=('B', 'G', 'R'), help="Mean BGR color value")
        parser.add_argument("-i", "--imgs_per_batch", default=2, type=int, help="Images per batch")
        parser.add_argument("-r", "--rois_per_image", default=64, type=int, help="ROIs per image. Use -1 for dynamic number of rois (e.g during test time)")
        parser.add_argument("-t", "--top_names", nargs='+', choices=top_names_choices, required=True,
                            type=str, help="ordered list of top names e.g input_image azimuth shape")
        parser.add_argument("-c", "--crop_target", default='bbx_visible',
                            choices=['bbx_amodal', 'bbx_visible'], type=str, help="bbx type used for cropping (Defaults to bbx_visible)")
        parser.add_argument("-sr", "--size_range", nargs=2, default=default_size_range, type=int,
                            metavar=('LOW', 'HI'), help="upper and lower bounds for length of shorter size of image in pixels")
        parser.add_argument("-sm", "--size_max", default=3072, type=int,  help="Max pixel size of the longest side of a scaled input image")
        parser.add_argument("-f", "--flip_ratio", default=0.5, type=float, help="Flip ratio in range [0, 1] (Defaults to 0.5)")
        parser.add_argument("-j", "--jitter_iou_min", default=0.7, type=float, help="Minimum jitter IOU for crop_target")

        params = parser.parse_args(param_str.split())

        print "---------------- FastRCNNDataLayer Config ----------------------"
        for arg in vars(params):
            print "\t{} \t= {}".format(arg, getattr(params, arg))
        print "------------------------------------------------------------"

        return params

    def setup(self, bottom, top):
        print "Setting up FastRCNNDataLayer ..."

        # params is expected as argparse style string
        params = self.parse_param_str(self.param_str)

        # Store the ordered list of top_names
        self.top_names = params.top_names
        # Store imgs_per_batch
        self.imgs_per_batch = params.imgs_per_batch
        # Store rois_per_image
        self.rois_per_image = params.rois_per_image
        # create mean bgr
        self.mean_bgr = np.array(params.mean_bgr)
        # set crop_target
        self.crop_target = params.crop_target
        # set size_range
        self.size_range = params.size_range
        # set size_max
        self.size_max = params.size_max
        # set flip_ratio
        self.flip_ratio = params.flip_ratio
        # set jitter_iou_min
        self.jitter_iou_min = params.jitter_iou_min

        assert len(top) == len(self.top_names), "Number of tops do not match specified top_names"

        assert self.size_range[1] >= self.size_range[0], "invalid size_range = {}".format(self.size_range)
        assert self.size_max >= self.size_range[1], "size_max needs to be greater than max shorter size"

        assert 0.0 < self.jitter_iou_min <= 1.0, "jitter_iou_min needs to be in [0, 1], but got {}".format(self.jitter_iou_min)

        rois_per_batch = max(2 * self.rois_per_image, 1)

        # set top shapes
        top_shapes = {
            "input_image": (self.imgs_per_batch, 3, 768, 1024),   # Use dummy H W
            "roi": (rois_per_batch, 5),
            "viewpoint": (rois_per_batch, 3),
            "bbx_amodal": (rois_per_batch, 4),
            "bbx_crop": (rois_per_batch, 4),
            "center_proj": (rois_per_batch, 2),
        }

        # Reshape tops
        for top_index, top_name in enumerate(self.top_names):
            top[top_index].reshape(*top_shapes[top_name])

        # Create a loader to load the images.
        self.image_loader = BatchImageLoader(transpose=False)

        # Create placeholder for annotations
        self.data_samples = []

        print 'RCNNDataLayer has been setup.'

    def add_dataset(self, dataset):
        """Add annotations from a json dataset"""
        print '---- Adding data from {} datatset -----'.format(dataset.name())
        print 'Number of data points (annotations) = {:,}'.format(len(self.data_samples))

        image_files = []
        image_infos = []
        for annotation in dataset.annotations():
            if 'objects' not in annotation or not annotation['objects']:
                continue

            image_file = osp.join(dataset.rootdir(), annotation['image_file'])
            image_files.append(image_file)

            image_info = {}
            for field in ['image_size', 'image_intrinsic']:
                if field in annotation:
                    image_info[field] = np.array(annotation[field])

            obj_infos = []
            for anno_obj in annotation['objects']:
                obj_info = {}
                obj_info['id'] = anno_obj['id']
                obj_info['category'] = anno_obj['category']

                for field in ['bbx_visible', 'bbx_amodal', 'viewpoint', 'center_proj', 'dimension']:
                    if field in anno_obj:
                        obj_info[field] = np.array(anno_obj[field])

                obj_infos.append(obj_info)

            image_info['objects'] = obj_infos
            image_infos.append(image_info)

        assert len(image_files) == len(image_infos)
        self.data_samples.extend(image_infos)
        self.image_loader.preload_images(image_files)

        print 'Number of images = {:,}'.format(len(self.data_samples))
        print "--------------------------------------------------------------------"

    def verify_data(self):
        """Verify all data"""
        print "Verifying data ...",
        assert len(self.data_samples) == len(self.image_loader), "len(self.data_samples) = {} while len(self.image_loader) = {}".format(
            len(self.data_samples), len(self.image_loader))
        for image_id, image_info in enumerate(self.data_samples):
            if 'image_size' in image_info:
                image_size = self.image_loader[image_id].shape[:2][::-1]
                assert np.all(image_info['image_size'] == image_size)

            if 'image_intrinsic' in image_info:
                assert image_info['image_intrinsic'].shape == (3, 3)

            for obj_info in image_info['objects']:
                if 'viewpoint' in obj_info:
                    vp = obj_info['viewpoint']
                    assert (vp >= -np.pi).all() and (vp < np.pi).all(), "Bad viewpoint = {}".format(vp)
        print "Done"

    def generate_datum_ids(self):
        """generate data ids"""
        # verify the data
        self.verify_data()

        # number of image_infos
        num_of_data_points = len(self.data_samples)

        # set of data indices in [0, num_of_data_points)
        self.data_ids = range(num_of_data_points)
        self.curr_data_ids_idx = 0

        assert len(self.data_ids) == num_of_data_points

        # Shuffle from the begining if in the train phase
        if self.phase == caffe.TRAIN:
            shuffle(self.data_ids)

        assert len(self.data_ids) > self.imgs_per_batch, 'imgs_per_batch ({})is smaller than total number of images ({}).'.format(
            self.imgs_per_batch, len(self.data_ids))
        print 'Total number of images = {:,}'.format(num_of_data_points)

        num_of_objects = sum([len(image_info['objects']) for image_info in self.data_samples])
        print 'Total number of objects = {:,}'.format(num_of_objects)
        return num_of_data_points

    def forward(self, bottom, top):
        """
        Load current batch of data and labels to caffe blobs
        """

        assert hasattr(self, 'data_ids'), 'Most likely data has not been initialized before calling forward()'

        image_ids = []
        for _ in xrange(self.imgs_per_batch):
            # Did we finish an epoch?
            if self.curr_data_ids_idx == len(self.data_ids):
                self.curr_data_ids_idx = 0
                shuffle(self.data_ids)

            # Current Data index
            img_idx = self.data_ids[self.curr_data_ids_idx]
            image_ids.append(img_idx)

            self.curr_data_ids_idx += 1

        mb_blobs = self.prepare_mini_batch(image_ids)

        for key, mb_blob in mb_blobs.iteritems():
            if key in self.top_names:
                top_index = self.top_names.index(key)
                top[top_index].reshape(*mb_blob.shape)
                top[top_index].data[...] = mb_blob

    def prepare_mini_batch(self, image_ids):
        mb_blobs = {}
        mb_blobs['input_image'], img_scales, img_flippings = self.prepare_image_blob(image_ids)
        mb_blobs.update(self.pepare_object_blobs(image_ids, img_scales, img_flippings))
        return mb_blobs

    def pepare_object_blobs(self, image_ids, img_scales, img_flippings):
        num_images = len(image_ids)
        assert len(img_scales) == len(img_flippings) == num_images

        obj_infos = []
        for i in xrange(num_images):
            image_id = image_ids[i]
            image_scale = img_scales[i]
            image_flip = img_flippings[i]

            assert isinstance(image_scale, (float))
            assert isinstance(image_flip, (bool))

            image_info = self.data_samples[image_id]
            obj_infos_curr_batch = sample_object_infos(image_info['objects'], self.rois_per_image, self.jitter_iou_min)
            for obj_info in obj_infos_curr_batch:
                if image_flip:
                    W = image_info['image_size'][0]
                    obj_info['bbx_crop'][[0, 2]] = W - obj_info['bbx_crop'][[2, 0]]
                    obj_info['bbx_amodal'][[0, 2]] = W - obj_info['bbx_amodal'][[2, 0]]
                    obj_info['center_proj'][0] = W - obj_info['center_proj'][0]
                    obj_info['viewpoint'][[0, 2]] = -obj_info['viewpoint'][[0, 2]]

                obj_info['roi'] = np.append(i, obj_info['bbx_crop'] * image_scale)
            obj_infos.extend(obj_infos_curr_batch)

        num_of_objects = len(obj_infos)
        object_blobs = {}
        object_blobs['roi'] = np.empty((num_of_objects, 5), dtype=np.float32)
        object_blobs['bbx_crop'] = np.empty((num_of_objects, 4), dtype=np.float32)
        object_blobs['bbx_amodal'] = np.empty((num_of_objects, 4), dtype=np.float32)
        object_blobs['viewpoint'] = np.empty((num_of_objects, 3), dtype=np.float32)
        object_blobs['center_proj'] = np.empty((num_of_objects, 2), dtype=np.float32)

        for i, obj_info in enumerate(obj_infos):
            for field in ['roi', 'bbx_crop', 'bbx_amodal', 'viewpoint', 'center_proj']:
                object_blobs[field][i, ...] = obj_info[field]

        return object_blobs

    def prepare_image_blob(self, image_ids):
        # returns image blob and also scaling, flipping params for each image
        num_images = len(image_ids)
        images = []
        img_scales = []
        img_flippings = []
        for i in xrange(num_images):
            img = self.image_loader[image_ids[i]].astype(np.float32) - self.mean_bgr
            assert img.ndim == 3 and img.shape[2] == 3

            flip = True if random() < self.flip_ratio else False
            target_size = randint(self.size_range[0], self.size_range[1])

            if flip:
                img = img[:, ::-1, :]

            img, img_scale = scale_image(img, target_size, self.size_max)

            images.append(img)
            img_scales.append(img_scale)
            img_flippings.append(flip)

        max_hw = np.array([img.shape[:2] for img in images]).max(axis=0)
        img_blob = np.zeros((num_images, max_hw[0], max_hw[1], 3), dtype=np.float32)

        for i in xrange(num_images):
            img = images[i]
            img_blob[i, 0:img.shape[0], 0:img.shape[1], :] = img

        # Make axis order NCHW
        img_blob = img_blob.transpose((0, 3, 1, 2))

        return img_blob, img_scales, img_flippings
