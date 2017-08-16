import os.path as osp
from collections import OrderedDict
from datasets import Dataset


def load_viewpoint_and_crop_annotations(filename, name=None):
    """Loads viewpoint, cropping pattern annotations from a text file.
    Expects each line to have 13 fields separated by space
    <image_filename> <viewpoint> <bbx_amodal> <bbx_crop>
    <viewpoint>, <bbx_amodal>, and <bbx_crop> have 4 numbers each.
    """
    assert osp.exists(filename), 'Path does not exist: {}'.format(filename)
    dataset = Dataset(filename if name is None else name)
    dataset.set_rootdir(osp.dirname(filename))

    print 'Loading annotations from {}'.format(filename)

    with open(filename, 'rb') as f:
        for line in f:
            line = line.rstrip('\n')
            tokens = line.split(' ')
            assert len(tokens) == 13

            annotation = OrderedDict()
            annotation['image_file'] = tokens[0]
            annotation['viewpoint'] = [float(x) for x in tokens[1:5]]
            annotation['bbx_amodal'] = [float(x) for x in tokens[5:9]]
            annotation['bbx_crop'] = [float(x) for x in tokens[9:13]]

            # Make sure a,e,t are in [0, 360)
            for i in xrange(3):
                annotation['viewpoint'][i] = annotation['viewpoint'][i] % 360.0

            dataset.add_annotation(annotation)

    print 'Loaded {} annotations from {}'.format(dataset.num_of_annotations(), filename)
    return dataset
