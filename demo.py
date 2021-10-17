import sys

import pandas as pd
import numpy.random as random
import torch
import torch.nn
import Polygon as plg
import argparse
from code.models.msra_resnet import get_pose_net
from code.models.Hourglass import get_large_hourglass_net
from code.tools.Diffusion import Diffusion
from config.Totaltext import config as cfg
from code.tools.augmentation import Normalize,Resize,Compose,BaseTransform
from code.tools.Detect import topK
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
import os
from code.tools.Detect import local_max
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io as io
import cv2
from code.tools.utils import shrink,get_centerpoints,draw_umich_gaussian,draw_dense_reg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from code.dataset import TextInstance
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


def generate_label(map_size,polygons):
    H = map_size
    W = map_size
    # 1.1得到文本区域的0 1 编码
    tr_mask = np.zeros((H, W), np.uint8)
    # 1.2得到文本区域的mask，如果文本是未表明文字的，就mask掉
    train_mask = np.ones((H, W), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
        if polygon.text == '#':
            cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))

    # 2.得到文本中心点
    # 6.得到文本中心点offsets
    center_points = []
    index_of_ct = np.zeros((cfg.max_annotation), dtype=np.int64)
    offsets_mask = np.zeros((cfg.max_annotation), dtype=np.uint8)
    center_offsets = np.zeros((cfg.max_annotation, 2), dtype=np.float32)
    for i in range(len(polygons)):

        center_x = polygons[i].center[0]
        center_y = polygons[i].center[1]
        center_points.append([int(center_x), int(center_y)])
        index_of_ct[i] = int(center_y) * W + int(center_x)
        if index_of_ct[i] < 0 or index_of_ct[i] >= H * W:
            # print("error")
            index_of_ct[i] = 0
        center_offsets[i][0] = int(center_x) - center_x
        center_offsets[i][1] = int(center_y) - center_y

        offsets_mask[i] = 1
    # 3.得到四个方向的位置编码
    geo_map = np.zeros((4, H, W), np.float32)
    # 4.得到四个方向的最大距离
    dense_wh = np.zeros((4, H, W))
    geo_max_dis = np.zeros((cfg.max_annotation, 4), dtype=np.float32)
    dense_wh_mask = np.stack([train_mask, train_mask, train_mask, train_mask])
    # print(dense_wh_mask.shape)
    for k in range(len(polygons)):
        m = np.zeros((4, H, W), np.float32)
        score_map = np.zeros((H, W), np.int32)
        cv2.fillPoly(score_map, [polygons[k].points.astype(np.int32)], color=(1, 0))
        mmax = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        for i in range(H):
            for j in range(W):
                dist = cv2.pointPolygonTest(polygons[k].points.astype(int), [i, j], False)
                if dist < 0:
                    continue

                # 上 x=x y=y-1 左
                x = i
                y = j
                while (y >= 0 and score_map[y][x]):
                    y = y - 1
                m[0][j][i] = np.abs(y - j)
                if mmax[0] < m[0][j][i]:
                    mmax[0] = m[0][j][i]
                # 下 x=x y=y+1 右
                x = i
                y = j
                while (y < W and score_map[y][x]):
                    y = y + 1
                m[1][j][i] = np.abs(y - j)
                if mmax[1] < m[1][j][i]:
                    mmax[1] = m[1][j][i]
                # 左 x=x-1 y=y 上
                x = i
                y = j
                while (x >= 0 and score_map[y][x]):
                    x = x - 1
                m[2][j][i] = np.abs(x - i)
                if mmax[2] < m[2][j][i]:
                    mmax[2] = m[2][j][i]
                # 右 x=x+1 y=y 下
                x = i
                y = j
                while (x < H and score_map[y][x]):
                    x = x + 1
                m[3][j][i] = np.abs(x - i)
                if mmax[3] < m[3][j][i]:
                    mmax[3] = m[3][j][i]

        for i in range(H):
            for j in range(W):
                dist = cv2.pointPolygonTest(polygons[k].points.astype(int), [i, j], False)
                if dist < 0:
                    continue

                m[0][j][i] = float(m[0][j][i]) / mmax[0]
                m[1][j][i] = float(m[1][j][i]) / mmax[1]
                m[2][j][i] = float(m[2][j][i]) / mmax[2]
                m[3][j][i] = float(m[3][j][i]) / mmax[3]
                for tt in range(4):
                    temp = min(m[tt][j][i], geo_map[tt][j][i]) if m[tt][j][i] > 0 and geo_map[tt][j][i] > 0 else \
                    m[tt][j][i]
                    geo_map[tt][j][i] = temp
        geo_max_dis[k] = mmax
    # 5.得到中心点heatmap
    heatmap = np.zeros((H, W), np.float32)
    for k in range(len(polygons)):
        rect = cv2.minAreaRect(polygons[k].points.astype(np.int32))
        box = cv2.boxPoints(rect)
        area = plg.Polygon(box).area()
        # radius=int(np.sqrt(area))
        radius = int(min(np.sqrt(area), 1.5 * np.abs(
            cv2.pointPolygonTest(polygons[k].points.astype(np.int32), center_points[k], True)))) + 1
        draw_umich_gaussian(heatmap, center_points[k], radius)
        draw_dense_reg(dense_wh, heatmap, center_points[k], geo_max_dis[k], radius)

    return tr_mask, train_mask, geo_map, index_of_ct, heatmap, dense_wh, center_offsets, dense_wh_mask, offsets_mask


def parse_mat( mat_path):
    annotation = io.loadmat(mat_path)
    polygons = []
    for cell in annotation['polygt']:
        if len(cell) <= 3:
            continue
        x = cell[1][0]
        y = cell[3][0]

        if len(cell) >= 5 and len(cell[4]) > 0:
            text = cell[4][0]
        else:
            text = ''
        try:
            ori = cell[5][0]
        except:
            ori = 'c'
        points = np.stack([x, y]).T.astype(np.int32)
        polygons.append(TextInstance(points, ori, text))

    return polygons

def display_gt(transform,image,anno_dir):
    if anno_dir== None:
        return
    polygons = parse_mat(anno_dir)
    for i in range(len(polygons)):
        polygons[i].find_centerline(15)
    _,polygons=transform(image,polygons)
    tr_mask, train_mask, geo_map, index_of_ct, heatmap, dense_wh, center_offsets, dense_wh_mask, offsets_mask =generate_label(cfg.input_size//cfg.downsample,polygons)
    ret = { 'input': image, 'hm': heatmap[np.newaxis, :]*255, 'train_mask': train_mask,'tr_mask':tr_mask,
           'hm_t': (geo_map[0])[np.newaxis, :], 'hm_b': (geo_map[1])[np.newaxis, :], 'hm_l': (geo_map[2])[np.newaxis, :],
           'hm_r': (geo_map[3])[np.newaxis, :],'dense_wh':dense_wh[np.newaxis,:]
           }
    plt.imshow(ret['input'])
    plt.show()
    plt.imshow(ret['tr_mask'].squeeze())
    plt.show()
    plt.imshow(ret['hm'].squeeze())
    plt.show()
    plt.imshow(ret['hm_t'].squeeze())
    plt.show()
    plt.imshow(ret['hm_b'].squeeze())
    plt.show()
    plt.imshow(ret['hm_l'].squeeze())
    plt.show()
    plt.imshow(ret['hm_r'].squeeze())
    plt.show()
    print(ret['dense_wh'].shape)
    plt.imshow(ret['dense_wh'].squeeze()[0])
    plt.show()
    plt.imshow(ret['dense_wh'].squeeze()[1])
    plt.show()
    plt.imshow(ret['dense_wh'].squeeze()[2])
    plt.show()
    plt.imshow(ret['dense_wh'].squeeze()[3])
    plt.show()

    plt.imshow(ret['hm'].squeeze())
    plt.show()
    '''
    temp = np.zeros((128, 128), np.uint8)
    for i in range(len(center_points)):
        cv2.circle(temp, (int(center_points[i][0]), int(center_points[i][1])), 5, 255, -2)
    cv2.imshow('center_points',temp)
    cv2.imshow("image_t",ret['hm_t'][0])
    cv2.waitKey(0)
    cv2.imshow('pred', temp)
    plt.imshow(ret['hm'].squeeze())
    plt.show()

    plt.imshow(ret['hm_t'].squeeze())
    plt.show()'''


def inference(maps):
    #heatmap=_sigmoid(maps['hm']).detach().cpu().numpy()[0][0]
    heatmap=local_max(_sigmoid(maps['hm']))#.detach().cpu().numpy()[0][0]
    #print(_sigmoid(maps['hm']).max())
    #tr = _sigmoid(maps['tr']).detach().cpu().numpy()[0][0]
    hm_t=_sigmoid(maps['hm_t']).detach().cpu().numpy()[0][0]
    hm_b = _sigmoid(maps['hm_b']).detach().cpu().numpy()[0][0]
    hm_l = _sigmoid(maps['hm_l']).detach().cpu().numpy()[0][0]
    hm_r = _sigmoid(maps['hm_r']).detach().cpu().numpy()[0][0]
    tblr=(_sigmoid(maps['dense_wh'])).detach().cpu().numpy()[0]
    print(tblr.shape)
    print(_sigmoid(maps['hm']).cpu().detach().numpy()[0][0].shape)
    plt.imshow(_sigmoid(maps['hm']).cpu().detach().numpy()[0][0])
    plt.show()
    plt.imshow(hm_t)
    plt.show()
    plt.imshow(hm_b)
    plt.show()
    plt.imshow(hm_l)
    plt.show()
    plt.imshow(hm_r)
    plt.show()
    #plt.imshow( heatmap.cpu().detach().numpy()[0][0])
    #cv2.waitKey(0)

    top_x, top_y = topK(heatmap,K=cfg.K)
    step_sizes=np.zeros((len(top_x),4))
    for i in range(len(top_x)):
        center_mask=np.zeros((128,128),np.uint8)
        cv2.circle(center_mask,[int(top_x[i]),int(top_y[i])],2,1,-1)
        #print(np.sum(center_mask),np.sum(tblr*center_mask))
        for j in range(4):
            step_sizes[i,j]=(np.sum(tblr[j]*center_mask))/(np.sum(center_mask))
    '''plt.imshow(_sigmoid(maps['hm']).detach().cpu().numpy()[0][0])
    plt.show()

    plt.imshow(tblr[0])
    plt.show()

    plt.imshow(hm_t)
    plt.show()
    plt.imshow(hm_b)
    plt.show()
    plt.imshow(hm_l)
    plt.show()
    plt.imshow(hm_r)
    plt.show()'''


    #cv2.waitKey(0)
    instances=[]
    for i in range(len(top_x)):
        instances.append((Diffusion(step_sizes[i],int(top_x[i]), int(top_y[i]), hm_t, hm_b, hm_l, hm_r)))
    i=0


    while i<cfg.max_diffusion:

        i = i + 1
        points_x = []
        points_y = []
        nums = []
        stop_count=0

        for j in range(len(top_x)):
            print(len(top_x),stop_count)
            if instances[j].walk_flag==False:
                stop_count+=1
                points_x.extend(instances[j].x_values)
                points_y.extend(instances[j].y_values)
                continue
            nums.append(instances[j].fill_walk())
            points_x.extend(instances[j].x_values)
            points_y.extend(instances[j].y_values)
        if stop_count == len(top_x):
            break
        plt.xlim((0, 128))
        plt.ylim((0, 128))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(i)

        plt.scatter(points_x, points_y, cmap='Blues', edgecolor='none')
        #plt.scatter(148, 176, c='green', edgecolors='none', s=100)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
        ax.invert_yaxis()
        plt.show()
    polygons=[]
    #temp=np.zeros((128,128),np.uint8)
    for i in range(len(top_x)):
        polygons.append(np.array(instances[i].hull.squeeze()).astype(np.int32))
        #cv2.fillPoly(temp,[np.array(instances[i].hull.squeeze()).astype(np.int32)],color=255)

    '''cv2.imshow('preds',temp*255)
        cv2.waitKey(0)'''
    return polygons



def demo(data_dir,modelfile,anno_dir):

    checkpoint = torch.load(modelfile)
    if cfg.model=='ResNet':
        model = get_pose_net(18, cfg.heads, 64).cuda()
    else:
        model=get_large_hourglass_net(18, cfg.heads, 64,checkpoint).cuda()
    #model.load_state_dict(checkpoint)
    model.eval()
    image=Image.open(data_dir)
    image_=np.array(image)
    plt.imshow(cv2.resize(image_,(128,128)))
    plt.show()

    transform=Compose([
        Resize(cfg.input_size),
        Normalize(cfg.means,cfg.stds)
    ])

    image,_=transform(image_)
    image = image.transpose(2, 0, 1)
    image=image[np.newaxis,:]
    input_image=torch.from_numpy(image).cuda()
    gt_transform = BaseTransform(size=cfg.input_size // cfg.downsample, mean=cfg.means,std=cfg.stds) if cfg.downsample > 0 else None
    display_gt(gt_transform, image_, anno_dir)
    model.eval()
    output=model(input_image)

    polygons_of_01=inference(output[0])
    temp=np.zeros((128,128),np.uint8)
    temp_=np.zeros((128,128),np.uint8)
    for i in range(len(polygons_of_01)):
        cv2.polylines(temp_, [polygons_of_01[i]], 1, 255)
        cv2.fillPoly(temp,[polygons_of_01[i]],color=255)
    plt.xlim((0, 128))
    plt.ylim((0, 128))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("final result")
    plt.imshow(temp)

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()
    plt.show()

    plt.imshow(temp_)

    ax = plt.gca()
    #ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    #ax.invert_yaxis()
    plt.show()











if __name__=='__main__':
    # ---------------- ARGS AND CONFIGS ----------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelfile', type=str,
                        default="/home/pei_group/jupyter/Wujingjing/Text_Detection/save_models/Hourglass_epoch55_71.7055.pt")
    parser.add_argument('--data_root', type=str, default="/home/pei_group/jupyter/Wujingjing/data/totaltext/Images/Test/img96.jpg")
    parser.add_argument('--anno_dir', type=str,
                        default="/home/pei_group/jupyter/Wujingjing/data/totaltext/gt/Test/poly_gt_img96.mat")

    opt = parser.parse_args()
    print("--- TRAINING ARGS ---")
    print(opt)

    demo(opt.data_root,opt.modelfile,opt.anno_dir)
