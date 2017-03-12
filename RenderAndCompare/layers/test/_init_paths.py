import os.path as osp
import sys

"""
Add  RenderAndCompare python module to python system search path i.e. PYTHONPATH
"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add root_dir to PYTHONPATH
root_dir = osp.join(osp.dirname(__file__), '..', '..', '..')
add_path(root_dir)
