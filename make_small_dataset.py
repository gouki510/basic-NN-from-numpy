from glob import glob
import os
import random
import cv2

data_path = "fashion_mnist"
phease = ["train","test"]
os.makedirs("small_dataset",exist_ok=True)
for p in phease:
    class_path = os.path.join(data_path,p)
    class_list = glob(os.path.join(class_path,'*'))
    for c in class_list:
        imgs = glob(os.path.join(c,"*"))
        img = random.sample(imgs,100)
        for i,im in enumerate(img):
            im = cv2.imread(im)
            path = os.path.join("small_dataset",p,c)
            os.makedirs(path,exist_ok=True)
            cv2.imwrite(os.path.join(path,"%04d.png"%i),im)
