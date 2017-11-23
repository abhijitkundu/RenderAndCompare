#!/usr/bin/env python
import os.path as osp
from os import system

BASE_DIR = osp.dirname(osp.abspath(__file__))
MATLAB_EXEC = 'matlab'  # !! MODIFY if necessary

def test_avp_nv_from_pred_view_file_list(cls_names, pred_view_file_list, det_bbox_mat_file_list, result_folder):
    '''
    @brief:
        evaluation of joint localization and viewpoint estimation (azimuth only)
        metric is AVP-NV
    @input:
        cls_names - a list of strings of PASCAL3D class names
        pred_view_file_list - a list of filenames of the predicted viewpoints
        det_bbox_mat_file_list - a list of det bbox file (contains boxes of Nx5, see combine_bbox_view.m for more)
        result_folder - result .mat files will be saved there
    @output:
        output .mat results (see combine_bbox_view.m for more details) into result_folder.
        display AVP-NV results
    @dependency:
        combine_bbox_view.m (output <clsname>_v<NV>.mat like chair_v8.mat)
        test_det.m  (assumes .mat files available in folder)
    '''
    C = len(cls_names)
    assert(C == len(pred_view_file_list))
    assert(C == len(det_bbox_mat_file_list))

    # if not osp.exists(result_folder):
    #     os.mkdir(result_folder)

    for i in xrange(C):
        # combine det and view
        matlab_cmd = "addpath('%s'); combine_bbox_view('%s','%s','%s','%s', %d);" % (
            BASE_DIR, cls_names[i], det_bbox_mat_file_list[i], pred_view_file_list[i], result_folder, 0)
        print matlab_cmd
        system('%s -nodisplay -r "try %s ; catch; end; quit;"' % (MATLAB_EXEC, matlab_cmd))

    # compute AVP-NV for all classes
    matlab_cmd = "addpath('%s'); test_det('%s');" % (BASE_DIR, result_folder)
    print matlab_cmd
    system('%s -nodisplay -r "try %s ; catch; end; quit;"' % (MATLAB_EXEC, matlab_cmd))

def main():
    cls_names = ['car']
    result_folder = "./"
    det_bbox_mat_file_list = [osp.join(result_folder, name + '_dets.mat') for name in cls_names]
    pred_view_file_list = [osp.join(result_folder, name + '_pred_view.txt') for name in cls_names]

    print 'det_bbox_mat_file_list=', det_bbox_mat_file_list
    print 'pred_view_file_list=', pred_view_file_list

    assert all(osp.isfile(fname) for fname in det_bbox_mat_file_list), 'Some files are missing'
    assert all(osp.isfile(fname) for fname in pred_view_file_list), 'Some files are missing'

    test_avp_nv_from_pred_view_file_list(cls_names, pred_view_file_list, det_bbox_mat_file_list, result_folder)


if __name__ == '__main__':
    main()
