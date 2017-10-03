"""
Some usefule dataset related functionality
"""

from copy import deepcopy
from random import shuffle

import cv2
import numpy as np

from RenderAndCompare.geometry import bbx_iou_overlap, create_jittered_bbx


def sample_object_infos(object_infos, number_of_objects, jitter_iou_min):
    """
    Sample number_of_objects object infos from input object_infos by jittering
    If number_of_objects < 0 then return the original object_infos
    """
    number_of_gt_objects = len(object_infos)
    assert number_of_gt_objects > 0, "Cannot have 0 objects"

    sampled_object_infos = []
    #if number_of_objects <= 0 (dynamic roi) then return the original object_infos
    if number_of_objects <= 0:
        for obj_id, oi in enumerate(object_infos):
            obj_info = deepcopy(oi)
            if jitter_iou_min < 1.0:
                obj_info['bbx_crop'] = create_jittered_box_with_no_conflicts(obj_id, object_infos, jitter_iou_min)
            else:
                obj_info['bbx_crop'] = oi['bbx_visible'].copy()
            sampled_object_infos.append(obj_info)
        return sampled_object_infos

    # assert 0.0 < jitter_iou_min <= 1.0, "For non-dynamic rois jitter_iou_min needs to be in [0, 1], but got {}".format(jitter_iou_min)

    obj_ids = range(number_of_gt_objects)
    shuffle(obj_ids)
    i = 0
    while len(sampled_object_infos) < number_of_objects:
        if i >= number_of_gt_objects:
            shuffle(obj_ids)
            i = 0
        obj_id = obj_ids[i]
        i += 1
        obj_info = deepcopy(object_infos[obj_id])
        if jitter_iou_min < 1.0:
            obj_info['bbx_crop'] = create_jittered_box_with_no_conflicts(obj_id, object_infos, jitter_iou_min)
        else:
            obj_info['bbx_crop'] = object_infos[obj_id]['bbx_visible'].copy()
        sampled_object_infos.append(obj_info)
    return sampled_object_infos

def create_jittered_box_with_no_conflicts(obj_id, object_infos, jitter_iou_min):
    """Jitter bbx by makin sure the jittered bbx still has the maximum overlap with original bbx"""
    bbx_crop_gt = np.array(object_infos[obj_id]['bbx_visible'])
    while True:
        bbx_crop = create_jittered_bbx(bbx_crop_gt, jitter_iou_min)
        max_iou_obj_id = np.asarray([bbx_iou_overlap(bbx_crop, np.array(object_info['bbx_visible'])) for object_info in object_infos]).argmax()
        if max_iou_obj_id == obj_id:
            break
    return bbx_crop



def draw_bbx2d(image, boxes, color=(0, 255, 0), thickness=1, copy=True):
    """Draw 2d boxes on image"""
    if copy:
        image_ = image.copy()
    else:
        image_ = image
    for box in boxes:
        bbx = np.array(box, dtype=np.float32)
        cv2.rectangle(image_, tuple(bbx[:2]), tuple(bbx[2:]), color, thickness)
    return image_
