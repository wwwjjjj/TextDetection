import numpy as np
import cv2
import torch
import torch.nn as nn

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
    #print(heatmap.shape)
    _,height,width=heatmap.size()
    topk_score,topk_ind=torch.topk(heatmap.view(1,-1),K)
    # topk_score:[1,K]      topk_ind:[1,K]
    topk_xs = (topk_ind % width).int().float()
    topk_ys = (topk_ind / width).int().float()
    #print(topk_xs[0].tolist(),topk_ys[0].tolist())
    return topk_xs[0].tolist(),topk_ys[0].tolist()




def local_max(heat,kernel=5):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
def get_center_points():

    pass
def inference(pred_maps):
    heatmap=pred_maps['hm']
    hm_t=pred_maps['hm_t']
    hm_b = pred_maps['hm_b']
    hm_l = pred_maps['hm_l']
    hm_r = pred_maps['hm_r']



    pass