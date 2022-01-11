import numpy as np
from model import *
from util import SGD
from omegaconf import OmegaConf
from data_module import Mydataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


def train(config):
    # model parameter define 
    img_size = config.img_size
    in_dim = img_size**2
    mid_dim = config.mid_dim
    mid_dim2 = config.mid_dim2
    p_noise = config.p_noise
    optim = config.optim
    lr = config.lr
    epoch_num = config.epoch
    data_path = config.data_path
    data_num = config.data_num
    label_dic = config.label_dic
    class_num = len(label_dic)
    lam = config.lam
    ord_idx = config.ord_idx
    # model define
    if config.mid_num == 3:
        model = FC_net3(in_dim, mid_dim, class_num)
        model_name = str(p_noise)+'_'+str(lr)+'_'+str(epoch_num)+"_"+str(mid_dim)
        print("model name:",model_name)
    elif config.mid_num == 4:
        model = FC_net4(in_dim, mid_dim, mid_dim2, class_num)
        model_name = str(p_noise)+'_'+str(lr)+'_'+str(epoch_num)+"_"+str(mid_dim)+"_"+str(mid_dim2)
        print("model name:",model_name)
    elif config.mid_num == 8:
        model = FC_net8(in_dim, class_num, lam, ord_idx)
        model_name = str(p_noise)+'_'+str(lr)+'_'+str(epoch_num)+"_"+"deep"+str(8)
        print("model name:",model_name)
    # dataset define
    dataset = Mydataset(data_path,p_noise,img_size,class_num)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    print("start lerning")
    for epoch in range(epoch_num): 
        train_acc = 0
        test_acc = 0
        train_loss = 0
        test_loss = 0
        for i in range(data_num):
            #train
            img,label = dataset.make_batch(i,'train')
            label_idx = label_dic[label]
            #on-hot変換
            t = np.identity(class_num)[label_idx]
            pred = model.forward(img)
            train_loss += model.loss_f(pred,t)
            model.backward()
            optimizer = SGD(lr)
            #parameter update
            for layer in model.layers:
                optimizer.update(layer)
            if np.argmax(t)==np.argmax(pred):
                train_acc+=1
            #test
            img,label = dataset.make_batch(i,'test')
            label_idx = label_dic[label]
            t = np.identity(class_num)[label_idx]
            in_dim = img.shape[0]
            pred = model.forward(img)
            test_loss += model.loss_f(pred,t)
            model.backward()
            if np.argmax(t)==np.argmax(pred):
                test_acc+=1
            sys_loss = test_loss/(i+1)
        temp_ta = train_acc/data_num
        temp_va = test_acc/data_num
        train_acc_list.append(temp_ta)
        test_acc_list.append(temp_va)
        train_loss_list.append(train_loss/data_num)
        test_loss_list.append(test_loss/data_num)
        print(f"\rEpoch {epoch+1}/{epoch_num}\ntrain_acc:{temp_ta},train_loss:{train_loss/data_num},test_acc:{temp_va},test_loss:{test_loss/data_num}",end="")
    # modelの保存
    f = open(model_name+".pickle","wb")
    pickle.dump(model,f)
    f.close
    return train_loss_list,train_acc_list,test_loss_list,test_acc_list,model_name

def plot(tll,tal,vll,val,name):
    fig = plt.figure()
    acc = fig.add_subplot(121)
    acc.set_title('ACC')
    loss = fig.add_subplot(122)
    loss.set_title("Loss")
    acc.plot(tal,color='red',label="train")
    acc.plot(val,color="blue",label="val")
    hans, labs = acc.get_legend_handles_labels()
    acc.legend(handles=hans, labels=labs)
    loss.plot(tll,color="red",label="train")
    loss.plot(vll,color="blue",label="val")
    hans, labs = loss.get_legend_handles_labels()
    loss.legend(handles=hans, labels=labs)
    fig.savefig(name+".png")
    
if __name__=='__main__':
    config = OmegaConf.load('config.yaml')
    tll,tal,vll,val,name = train(config)
    plot(tll,tal,vll,val,name)        

