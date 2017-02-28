import os.path as osp
from collections import OrderedDict
import json


class Dataset(object):
    def __init__(self, name = 'RenderAndCompareDataset', data=None):
        if data is None:
            self.data = OrderedDict()
            self.data['name'] = name
            self.data['rootdir'] = ''
            self.data['meta'] = ''
            self.data['annotations'] = []
        else:
            self.data = data
        
        
    def data(self):
        return self.data
    
    def name(self):
        return self.data['name']
    
    def rootdir(self):
        return self.data['rootdir']
    
    def annotations(self):
        return self.data['annotations']

    def num_of_annotations(self):
        return len(self.data['annotations'])
    
    def set_name(self, new_name):
        self.data['name'] = new_name
    
    def set_rootdir(self, rootdir):
        self.data['rootdir'] = rootdir
    
    def add_annotation(self, annotation):
        self.data['annotations'].append(annotation)

    def write_data_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2, separators=(',', ':'))
        print 'Saved dataset to {}'.format(filename)

    def set_data_from_json(self, filename):
        '''Sets the dataset data from a JSON file'''
        with open(filename, 'r') as f:
            self.data = json.load(f, object_pairs_hook=OrderedDict)

    @classmethod
    def from_json(cls, filename):
        '''Constricts the dataset from a JSON file'''
        with open(filename, 'r') as f:
            loaded_data = json.load(f, object_pairs_hook=OrderedDict)
        return cls(data=loaded_data)

    def __repr__(self):
        return 'Dataset(name="%s", with %d annotations)' % (self.name(), self.num_of_annotations())




def loadPascal3Ddataset(filename, name=None):
    assert osp.exists(filename), 'Path does not exist: {}'.format(filename)
    dataset = Dataset(filename if name==None else name)
    dataset.set_rootdir(osp.dirname(filename))

    print 'Loading Pascal3Ddataset from {}'.format(filename)

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



