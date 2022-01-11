import cv2 
import numpy as np
import os
from glob import glob
import random

class Mydataset:
    def __init__(self,data_path,p_noise,img_size,class_num):
        self.data_path = data_path
        self.train_path = os.path.join(self.data_path,'train')
        self.test_path = os.path.join(self.data_path,'test')
        self.p_noise = p_noise
        self.img_size = img_size
        self.class_num = class_num
    
    def make_batch(self,batch_idx,phase):
        if phase=='train':
            label_list = glob(os.path.join(self.train_path,"*"))
            label = label_list[int(batch_idx%self.class_num)].split('/')[-1]
            img_path = random.sample(glob(os.path.join(self.train_path,label,'*.png')),1)[0]
            img = cv2.imread(img_path)
            img = cv2.resize(img,(self.img_size,self.img_size))
            h,w,_ = img.shape
            #noise処理
            pts_x = np.random.randint(0,h-1,int(w*h*self.p_noise))
            pts_y = np.random.randint(0,w-1,int(w*h*self.p_noise))
            #[0,1]の乱数
            img[(pts_x,pts_y)] = np.random.random_sample(3)
            #grayscale
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            h,w = img.shape
            img = np.ravel(img)
            # [0,255]-> [0,1]
            img = img/255
            return img,label

        if phase=='test':
            label_list = glob(os.path.join(self.test_path,"*"))
            label = random.sample(label_list,1)[0].split('/')[-1]
            img_path = random.sample(glob(os.path.join(self.test_path,label,'*.png')),1)[0]
            img = cv2.imread(img_path)
            img = cv2.resize(img,(self.img_size,self.img_size))
            h,w,_ = img.shape
            #noise処理
            pts_x = np.random.randint(0,h-1,int(w*h*self.p_noise))
            pts_y = np.random.randint(0,w-1,int(w*h*self.p_noise))
            #[0,1]の乱数
            img[(pts_x,pts_y)] = np.random.random_sample(3)
            #grayscale
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            h,w = img.shape
            img = np.ravel(img)
            # [0,255]-> [0,1]
            img = img/255
            return img,label
