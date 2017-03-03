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
            assert osp.exists(self.data['rootdir']), 'Root dir does not exist: {}'.format(self.data['rootdir'])
        
        
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