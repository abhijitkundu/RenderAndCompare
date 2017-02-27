#!/usr/bin/env python
import os.path as osp
import sys

# Add root dir to path
sys.path.insert(0, osp.join(osp.join(osp.dirname(__file__), '..', '..')))
import RenderAndCompare as rac

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='converts pascal3d annotations to json')
    parser.add_argument('annotation_file', help='Path to Annotation file to import')
    parser.add_argument("-o", "--outfile", required=True, help="Filename to save the imported annotations")
    args = parser.parse_args()


    dataset = rac.datasets.loadPascal3Ddataset(args.annotation_file)
    dataset.write_data_to_json(args.outfile)


