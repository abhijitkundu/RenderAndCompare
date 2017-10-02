"""
Some usefule dataset related functionality
"""

from random import shuffle
import numpy as np
import cv2
from RenderAndCompare.geometry import create_jittered_bbx, bbx_iou_overlap

def sample_object_infos(object_infos, number_of_objects, jitter_iou_min):
    """
    Sample number_of_objects object infos from input object_infos by jittering
    If number_of_objects < 0 then return the original object_infos
    """
    number_of_gt_objects = len(object_infos)
    assert number_of_gt_objects > 0, "Cannot have 0 objects"

    sampled_object_infos = []
    #if number_of_objects < 0 then return the original object_infos
    if number_of_objects < 0:
        for oi in object_infos:
            obj_info = oi.copy()
            obj_info['bbx_crop'] = oi['bbx_visible']
            sampled_object_infos.append(obj_info)
        return sampled_object_infos

    obj_ids = range(number_of_gt_objects)
    shuffle(obj_ids)
    i = 0
    while len(sampled_object_infos) < number_of_objects:
        if i >= number_of_gt_objects:
            shuffle(obj_ids)
            i = 0
        obj_id = obj_ids[i]
        i += 1
        bbx_crop_gt = np.array(object_infos[obj_id]['bbx_visible'])
        while True:
            bbx_crop = create_jittered_bbx(bbx_crop_gt, jitter_iou_min)
            max_iou_obj_id = np.asarray([bbx_iou_overlap(bbx_crop, np.array(object_infos[j]['bbx_visible'])) for j in xrange(number_of_gt_objects)]).argmax()
            if max_iou_obj_id == obj_id:
                break

        obj_info = object_infos[obj_id].copy()
        obj_info['bbx_crop'] = bbx_crop
        sampled_object_infos.append(obj_info)
    return sampled_object_infos

def draw_bbx2d(image, boxes, color=(0, 255, 0), thickness=1, copy=True):
    if copy:
        image_ = image.copy()
    else:
        image_ = image
    for box in boxes:
        bbx = np.array(box, dtype=np.float32)
        cv2.rectangle(image_, tuple(bbx[:2]), tuple(bbx[2:]), color, thickness)
    return image_
