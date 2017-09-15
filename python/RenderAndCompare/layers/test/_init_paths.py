"""
Add  RenderAndCompare python module to python system search path i.e. PYTHONPATH
"""
import os.path as osp
import sys


def add_path(path):
    """Add path to sys.path"""
    if path not in sys.path:
        sys.path.insert(0, path)


# Add root_dir to PYTHONPATH
parent_dir = osp.dirname(__file__)
root_dir = osp.join(parent_dir, '..', '..', '..')
add_path(root_dir)
