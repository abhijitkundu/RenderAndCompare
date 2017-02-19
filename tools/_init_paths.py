
import os.path
import sys

"""
Add lib paths to python system search path i.e. PYTHONPATH
"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

cur_dir = os.path.dirname(__file__)
root_dir = os.path.join(cur_dir, '..')

# Add lib to PYTHONPATH
add_path(os.path.join(root_dir, 'lib'))
