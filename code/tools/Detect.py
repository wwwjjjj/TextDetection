import numpy as np
import cv2
import torch
import torch.nn as nn
from shapely.geometry import Polygon

def _gather_feat(feat, ind, mask=None):
    print(feat.shape,ind.shape)
    dim  = feat.size(1)
    #ind  = ind.unsqueeze(2).expand(-1,  dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def topK(heatmap,K=40):
    heatmap=torch.squeeze(heatmap,dim=1)
    _,height,width=heatmap.size()
    #print(heatmap.size())
    topk_score,topk_ind=torch.topk(heatmap.view(1,-1),K)

    topk_ind=topk_ind[topk_score>=0.2]
    topk_xs = (topk_ind % width).int().float()
    topk_ys = (topk_ind / width).int().float()
    return topk_xs.tolist(),topk_ys.tolist()


def approx_area_of_intersection(det_x, det_y, gt_x, gt_y):
    """
    This helper determine if both polygons are intersecting with each others with an approximation method.
    Area of intersection represented by the minimum bounding rectangular [xmin, ymin, xmax, ymax]
    """
    det_ymax = np.max(det_y)
    det_xmax = np.max(det_x)
    det_ymin = np.min(det_y)
    det_xmin = np.min(det_x)

    gt_ymax = np.max(gt_y)
    gt_xmax = np.max(gt_x)
    gt_ymin = np.min(gt_y)
    gt_xmin = np.min(gt_x)

    all_min_ymax = np.minimum(det_ymax, gt_ymax)
    all_max_ymin = np.maximum(det_ymin, gt_ymin)

    intersect_heights = np.maximum(0.0, (all_min_ymax - all_max_ymin))

    all_min_xmax = np.minimum(det_xmax, gt_xmax)
    all_max_xmin = np.maximum(det_xmin, gt_xmin)
    intersect_widths = np.maximum(0.0, (all_min_xmax - all_max_xmin))

    return intersect_heights * intersect_widths
def iou(det_x, det_y, gt_x, gt_y):
    """
    This helper determine the intersection over union of two polygons.
    """

    if approx_area_of_intersection(det_x, det_y, gt_x, gt_y) > 1: #only proceed if it passes the approximation test
        ymax = np.maximum(np.max(det_y), np.max(gt_y)) + 1
        xmax = np.maximum(np.max(det_x), np.max(gt_x)) + 1
        bin_mask = np.zeros((ymax, xmax))
        det_bin_mask = np.zeros_like(bin_mask)
        gt_bin_mask = np.zeros_like(bin_mask)

        rr, cc = polygon(det_y, det_x)
        det_bin_mask[rr, cc] = 1

        rr, cc = polygon(gt_y, gt_x)
        gt_bin_mask[rr, cc] = 1

        final_bin_mask = det_bin_mask + gt_bin_mask

        #inter_map = np.zeros_like(final_bin_mask)
        inter_map = np.where(final_bin_mask == 2, 1, 0)
        inter = np.sum(inter_map)

        #union_map = np.zeros_like(final_bin_mask)
        union_map = np.where(final_bin_mask > 0, 1, 0)
        union = np.sum(union_map)
        return inter / float(union + 1.0)
        #return np.round(inter / float(union + 1.0), 2)
    else:
        return 0


def local_max(heat,kernel=5):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
def get_center_points():

    pass
