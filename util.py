import numpy as np 

class SGD:
    def __init__(self,lr):
        self.lr = lr
        
    def update(self,layer):
        layer.W -= self.lr*layer.dW
        layer.b -= self.lr*layer.db
    

