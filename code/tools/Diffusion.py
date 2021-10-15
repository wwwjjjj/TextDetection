import matplotlib.pyplot as plt
import numpy as np
import cv2
class Diffusion():
#漫步
    def __init__(self,x,y, m0,m1,m2,m3,num_points=5000):
        self.num_points = num_points
        # 所有随机漫步都始于(0,0)
        self.x_values = [x]#list(center_points[:, 0])
        self.y_values = [y]#list(center_points[:, 1])

        self.pre_points=[[x,y]]
        self.pre_area=0
        self.hull=None
        self.walk_flag=True

        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3


    def fill_walk(self):
        """计算随机漫步包含的所有点"""
        if self.walk_flag==False:
            return len(self.x_values)
        hull = cv2.convexHull(np.array(self.pre_points,np.float32), clockwise=True, returnPoints=True)
        area=cv2.contourArea(hull)
        #print(area, self.pre_area, "**")
        if self.pre_area!=0 and np.abs(area-self.pre_area)<=0.2*self.pre_area:

            self.hull=hull
            self.walk_flag=False
        else:
            self.pre_area=area
        destinations = [ [0, -1], [0, 1],
                        [-1, 0], [1, 0]]
        step = 10
        # print(self.x_values[-1])
        next_ax = []
        next_ay = []
        next_points=[]

        #print(len(self.pre_x))
        for i in range(len(self.pre_points)):
            value_x = self.pre_points[i][0]
            value_y = self.pre_points[i][1]
            for choose in range(4):
                if choose==0:
                    score=self.m0
                if choose==1:
                    score=self.m1
                if choose==2:
                    score=self.m2
                if choose==3:
                    score=self.m3
                deltax = score[int(value_y)][int(value_x)] * destinations[choose][0] * step
                deltay = score[int(value_y)][int(value_x)] * destinations[choose][1] * step
                '''if np.abs(deltax)<1e-2 and np.abs(deltay)<1e-2:
                continue'''
                next_x = value_x + deltax
                next_y = value_y + deltay
                next_ax.append(next_x)
                next_ay.append(next_y)
                next_points.append([next_x,next_y])


        self.pre_points.clear()
        self.pre_points.extend(next_points)
        #print(len(self.pre_x))
        self.x_values.extend(next_ax)

        self.y_values.extend(next_ay)
        return len(self.x_values)
