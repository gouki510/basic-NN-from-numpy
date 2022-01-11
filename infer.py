import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle
import argparse

class Infer:
    def __init__(self,img_dir,out_dir,weight_path):
        self.img_dir = img_dir
        self.img_list = glob(os.path.join(self.img_dir,"*"))
        self.weight_path = weight_path
        self.labels ={
            0:"Top",
            1:"Trouser",
            2:"Pullover",
            3:"Dress",
            4:"Coat",
            5:"Sandal",
            6:"Shirt",
            7:"Sneaker",
            8:"Bag",
            9:"Boot",
        }
        self.out_dir = out_dir
        os.makedirs(self.out_dir,exist_ok=True)

    def model_load(self):
        f = open(self.weight_path,'rb')
        model = pickle.load(f)
        return model

    def predict(self):
        model = self.model_load()
        for img_path in self.img_list:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray,(28,28))
            img_gray = cv2.bitwise_not(img_gray)
            img_flat = np.ravel(img_gray)
            # [0,255]-> [0,1]
            img_flat = img_flat/255
            pred = model.forward(img_flat)
            pred_label = self.labels[np.argmax(pred)]
            t_label = os.path.splitext(os.path.basename(img_path))[0]
            fig = plt.figure()
            raw_plot = fig.add_subplot(121)
            pred_plot = fig.add_subplot(122)
            raw_plot.imshow(img)
            pred_plot.imshow(img_gray,cmap = "gray")
            raw_plot.set_title(t_label)
            pred_plot.set_title(pred_label)
            file_name = os.path.join(self.out_dir,t_label)
            fig.savefig(file_name+".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference fashion category")
    parser.add_argument("-i","--img_dir",default='imgs')
    parser.add_argument("-o","--output_dir",default='output')
    parser.add_argument("-w","--weight_path",default="weight.pickle")
    args = parser.parse_args()
    img_dir = args.img_dir
    out_dir = args.output_dir
    weight_path = args.weight_path
    infer = Infer(img_dir,out_dir,weight_path)
    infer.predict() 