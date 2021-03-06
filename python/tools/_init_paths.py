import sys
import os.path as osp

"""
Add  RenderAndCompare python module to python system search path i.e. PYTHONPATH
"""

def add_path(path):
    """Add path to PYTHONPATH """
    if path not in sys.path:
        sys.path.insert(0, path)

# Add root_dir to PYTHONPATH
root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
add_path(osp.join(root_dir, 'python'))
