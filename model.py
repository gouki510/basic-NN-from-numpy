import numpy as np
from layer import Affine,Relu,softmax,cross_entropy_error

class FC_net3:
    def __init__(self,in_dim,mid_dim,out_dim):
        self.mid_dim = mid_dim
        self.out_dim = out_dim 
        self.w1 = np.random.randn(in_dim,mid_dim)
        self.b1 = np.random.randn(mid_dim)
        self.w2 = np.random.randn(mid_dim,out_dim)
        self.b2 = np.random.randn(out_dim)
        self.relu = Relu()
        self.fc1 = Affine(self.w1,self.b1)
        self.fc2 = Affine(self.w2,self.b2)
        self.softmax = softmax
        self.t = None
        self.y = None
        self.loss = None
        self.layers = [self.fc1,self.fc2]

    def forward(self,x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        x = self.softmax(x)
        self.y = x
        return self.y
    
    def loss_f(self,pred,t):
        self.t = t
        self.loss = cross_entropy_error(self.y,t)
        return self.loss

    def backward(self):
        dout = self.y - self.t
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)
        
class FC_net4:
    def __init__(self,in_dim,mid_dim,mid_dim2,out_dim):
        self.mid_dim = mid_dim
        self.mid_dim2 = mid_dim2
        self.out_dim = out_dim 
        self.w1 = np.random.randn(in_dim,mid_dim)
        self.b1 = np.random.randn(mid_dim)
        self.w2 = np.random.randn(mid_dim,mid_dim2)
        self.b2 = np.random.randn(mid_dim2)
        self.w3 = np.random.randn(mid_dim2,out_dim)
        self.b3 = np.random.randn(out_dim)
        self.relu = Relu()
        self.relu2 = Relu()
        self.fc1 = Affine(self.w1,self.b1)
        self.fc2 = Affine(self.w2,self.b2)
        self.fc3 = Affine(self.w3,self.b3)
        self.softmax = softmax
        self.t = None
        self.y = None
        self.loss = None
        self.layers = [self.fc1,self.fc2,self.fc3]

    def forward(self,x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        x = self.softmax(x)
        self.y = x
        return self.y
    
    def loss_f(self,pred,t):
        self.t = t
        self.loss = cross_entropy_error(self.y,t)
        return self.loss

    def backward(self):
        dout = self.y - self.t
        dout = self.fc3.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)


class FC_net8:
    def __init__(self,in_dim,out_dim,lam,ord_idx):
        self.out_dim = out_dim 
        # parameter intialize
        self.w1 = np.random.randn(in_dim,512)
        self.b1 = np.random.randn(512)
        self.w2 = np.random.randn(512,384)
        self.b2 = np.random.randn(384)
        self.w3 = np.random.randn(384,256)
        self.b3 = np.random.randn(256)
        self.w4 = np.random.randn(256,192)
        self.b4 = np.random.randn(192)
        self.w5 = np.random.randn(192,64)
        self.b5 = np.random.randn(64)
        self.w6 = np.random.randn(64,32)
        self.b6 = np.random.randn(32)
        self.w7 = np.random.randn(32,16)
        self.b7 = np.random.randn(16)
        self.w8 = np.random.randn(16,10)
        self.b8 = np.random.randn(10)

        self.relu = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()
        self.relu4 = Relu()
        self.relu5 = Relu()
        self.relu6 = Relu()
        self.relu7 = Relu()
        self.fc1 = Affine(self.w1,self.b1)
        self.fc2 = Affine(self.w2,self.b2)
        self.fc3 = Affine(self.w3,self.b3)
        self.fc4 = Affine(self.w4,self.b4)
        self.fc5 = Affine(self.w5,self.b5)
        self.fc6 = Affine(self.w6,self.b6)
        self.fc7 = Affine(self.w7,self.b7)
        self.fc8 = Affine(self.w8,self.b8)

        self.lam = lam
        self.ord = ord_idx

        self.softmax = softmax
        self.t = None
        self.y = None
        self.loss = None
        self.layers = [self.fc1,self.fc2,self.fc3,self.fc4,self.fc5,self.fc6,self.fc7,self.fc8]

    def forward(self,x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        x = self.relu3.forward(x)
        x = self.fc4.forward(x)
        x = self.relu4.forward(x)
        x = self.fc5.forward(x)
        x = self.relu5.forward(x)
        x = self.fc6.forward(x)
        x = self.relu6.forward(x)
        x = self.fc7.forward(x)
        x = self.relu7.forward(x)
        x = self.fc8.forward(x)
        x = self.softmax(x)
        self.y = x
        return self.y
    
    def loss_f(self,pred,t):
        self.t = t
        self.loss = cross_entropy_error(self.y,t)
        return self.loss

    def regularize(self,lam,ord_idx):
        penalty = 0
        for fc in  self.layers:
            penalty+=lam*np.linalg.norm(fc.W,ord=ord_idx)
        return penalty/len(self.layers)

    def backward(self):
        dout = self.y - self.t
        dout = self.fc8.backward(dout)
        dout = self.relu7.backward(dout)
        dout = self.fc7.backward(dout)
        dout = self.relu6.backward(dout)
        dout = self.fc6.backward(dout)
        dout = self.relu5.backward(dout)
        dout = self.fc5.backward(dout)
        dout = self.relu4.backward(dout)
        dout = self.fc4.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc3.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)


