# coding: utf-8
import numpy as np
from src.util import *


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):   
        self.mask = (x <= 0) # True False矩阵
        out = x.copy()
        out[self.mask] = 0   # 将self.mast中元素为True的位置设为0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  
        self.x = x                     

        out = np.dot(self.x, self.W) + self.b  

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape) 
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t) 
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:  # 标签label是类别
            dx = self.y.copy()                    
            dx[np.arange(batch_size), self.t] -= 1  
            dx = dx / batch_size                    
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # FN, C, FH, FW分别是滤波器输出通道数，c是图片channel，FH=FW是滤波器大小
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        self.cache = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        out = self._forward_fast(x)
        return out

    def backward(self, dout):
        dx = self._backward_fast(dout)
        return dx

    def _forward_fast(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape  # N是样本个数，C是通道数，H=W是图片大小
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)  # 输出图片大小
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # 图片转成矩阵,会填充重复值
        col_W = self.W.reshape(FN, -1).T                # 权重转矩阵，直接reshape

        out = np.dot(col, col_W) + self.b               # reshape后是N，再是高宽，再是通道数，要把通道数提前
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def _backward_fast(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
    def _forward_naive(self, x):
        w = self.W
        b = self.b
        conv_param = {'pad':self.pad, 'stride':self.stride}

        out, cache = conv_forward_naive(x, w, b, conv_param)
        self.cache = cache
        return out
    
    def _backward_naive(self, dout):
        cache  = self.cache
        dx, dw, db = conv_backward_naive(dout, cache)
        self.dW = dw
        self.db = db
        
        assert dx is not None, 'dx should not be none'
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
