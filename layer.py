import numpy as np


#全結合層
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
    #誤差逆伝播
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.reshape(-1,1), dout.reshape(1,-1))
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  
        return dx

#Relu
class Relu:
    def __init__(self):
        self.mask = None
#0以下は0,0以上はそのまま
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
def cross_entropy_error(y, t):
    return -np.sum(np.log(y + 1e-7)*t)





