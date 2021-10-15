import numpy as np
from torch.utils.data import Dataset
import os
import Polygon as plg
from code.tools.utils import pil_load_img,find_bottom,find_long_edges,draw_dense_reg,split_edge_seqence,norm2, shrink,get_centerpoints,draw_umich_gaussian,get_furthest_point_from_edge
import cv2
import scipy.io as io
from config.Totaltext import config as cfg
import copy
class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)



    def find_centerline(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text

        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        self.center=[center_points[int(len(center_points) / 2)][0],center_points[int(len(center_points) / 2)][1]]


        return center_points


    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)

class TotalText(Dataset):
    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None,map_transform=None,map_size=512):
        super().__init__()
        self.data_root = data_root
        self.is_training = is_training
        self.transform=transform
        self.map_transform=map_transform
        self.map_size=map_size

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))

        self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '').lstrip('0')) for img_name in
                                self.image_list]




    def generate_label(self,image,polygons):

        H = self.map_size
        W = self.map_size
        #1.1得到文本区域的0 1 编码
        tr_mask = np.zeros((H,W), np.uint8)
        #1.2得到文本区域的mask，如果文本是未表明文字的，就mask掉
        train_mask=np.ones((H,W),np.uint8)
        for polygon in polygons:
            cv2.fillPoly(tr_mask,[polygon.points.astype(np.int32)],color=(1,))
            if polygon.text=='#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))

        #2.得到文本中心点
        #6.得到文本中心点offsets
        center_points = []
        index_of_ct=np.zeros((cfg.max_annotation), dtype=np.int64)
        offsets_mask=np.zeros((cfg.max_annotation), dtype=np.uint8)
        center_offsets=np.zeros((cfg.max_annotation,2), dtype=np.float32)
        for i in range(len(polygons)):

            center_x = polygons[i].center[0]
            center_y = polygons[i].center[1]
            center_points.append([int(center_x), int(center_y)])
            index_of_ct[i]=int(center_y)*W+int(center_x)
            if index_of_ct[i]<0 or index_of_ct[i]>=H*W:
                #print("error")
                index_of_ct[i]=0
            center_offsets[i][0] = int(center_x) -center_x
            center_offsets[i][1] = int(center_y) -center_y

            offsets_mask[i]=1
        #3.得到四个方向的位置编码
        geo_map = np.zeros((4, H,W),np.float32)
        #4.得到四个方向的最大距离
        dense_wh = np.zeros((4,H,W))
        geo_max_dis=np.zeros((cfg.max_annotation,4),dtype=np.float32)
        dense_wh_mask=np.stack([train_mask,train_mask,train_mask,train_mask])
        #print(dense_wh_mask.shape)
        for k in range(len(polygons)):
            m = np.zeros((4, H, W), np.float32)
            score_map=np.zeros((H,W),np.int32)
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
                    while (y>=0 and score_map[y][x]):
                        y = y - 1
                    m[0][j][i] = np.abs(y - j)
                    if mmax[0] < m[0][j][i]:
                        mmax[0] = m[0][j][i]
                    # 下 x=x y=y+1 右
                    x = i
                    y = j
                    while (y<W and score_map[y][x]):
                        y = y + 1
                    m[1][j][i] = np.abs(y - j)
                    if mmax[1] < m[1][j][i]:
                        mmax[1] = m[1][j][i]
                    # 左 x=x-1 y=y 上
                    x = i
                    y = j
                    while (x>=0 and score_map[y][x]):
                        x = x - 1
                    m[2][j][i] = np.abs(x - i)
                    if mmax[2] < m[2][j][i]:
                        mmax[2] = m[2][j][i]
                    # 右 x=x+1 y=y 下
                    x = i
                    y = j
                    while (x<H and score_map[y][x]):
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
                        temp=min(m[tt][j][i],geo_map[tt][j][i]) if  m[tt][j][i]>0 and geo_map[tt][j][i]>0 else m[tt][j][i]
                        geo_map[tt][j][i]=temp
            geo_max_dis[k]=mmax
        #5.得到中心点heatmap
        heatmap = np.zeros((H, W), np.float32)
        for k in range(len(polygons)):
            rect = cv2.minAreaRect(polygons[k].points.astype(np.int32))
            box = cv2.boxPoints(rect)
            area = plg.Polygon(box).area()
            #radius=int(np.sqrt(area))
            radius = int(min(np.sqrt(area),1.5*np.abs(cv2.pointPolygonTest(polygons[k].points.astype(np.int32), center_points[k], True))))+1
            draw_umich_gaussian(heatmap, center_points[k], radius)
            draw_dense_reg(dense_wh, heatmap, center_points[k], geo_max_dis[k],radius)

        return tr_mask,train_mask,geo_map,index_of_ct,heatmap,dense_wh,center_offsets,dense_wh_mask,offsets_mask

    def parse_mat(self,mat_path):
        annotation=io.loadmat(mat_path)
        polygons=[]
        for cell in annotation['polygt']:
            if len(cell)<=3 or len(cell[1][0])<=3:
                continue
            x=cell[1][0]
            y=cell[3][0]

            if len(cell) >= 5 and len(cell[4]) > 0:
                text = cell[4][0]
            else:
                text = ''
            try:
                ori=cell[5][0]
            except:
                ori='c'
            points=np.stack([x,y]).T.astype(np.int32)
            polygons.append(TextInstance(points,ori,text))

        return polygons


    def __getitem__(self, item):
        #得到编码后的四个方向的map，中心点（heatmap），每个像素点和真实中心点之间的偏移量，每个像素点所在的文本区域的大概宽高
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        #print(image_id)

        # Read image data
        image = pil_load_img(image_path)

        H,W,_=image.shape
        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)

        polygons=self.parse_mat(annotation_path)
        for i in range(len(polygons)):
            polygons[i].find_centerline(15)

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))
        if self.map_transform:
            _,polygons=self.map_transform(image,copy.copy(polygons))


        tr_mask,train_mask,m,index_of_ct,heatmap,dense_wh,center_offsets,dense_wh_mask,offsets_mask=self.generate_label(image,polygons)

        #print(dense_wh.shape)

        image = image.transpose(2,0,1)  # 0, 3, 1, 2
        ret={'input':image,'hm':heatmap[np.newaxis,:],'trm':train_mask[np.newaxis,:],
             'hm_t':(m[0])[np.newaxis,:],'hm_b':(m[1])[np.newaxis,:],'hm_l':(m[2])[np.newaxis,:],'hm_r':(m[3])[np.newaxis,:],
             'dense_wh':dense_wh[np.newaxis,:],'offsets':center_offsets,'dense_wh_mask':dense_wh_mask,'off_mask':offsets_mask,
             'center_points':index_of_ct
             }


        return ret#image,tr_mask[np.newaxis,:],m,meta,shrink_mask[np.newaxis,:],heatmap[np.newaxis,:]



    def __len__(self):
        return len(self.image_list)
