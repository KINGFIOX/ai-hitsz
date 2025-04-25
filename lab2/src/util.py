# coding: utf-8
import numpy as np
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import urllib.request

def Todo():
    raise NotImplementedError

def sigmoid(x):
    """
    x: sigmoid函数的输入值,numpy数组类型
    返回:sigmoid函数值,numpy数组类型
    """
    # 指数函数可直接使用numpy.exp()
    Todo()

def sigmoid_grad(x):
    """ sigmoid函数梯度
    x: numpy数组类型
    返回: 梯度值,numpy数组类型
    """
    Todo()


def softmax(x):
    """ 
    x: numpy数组类型,需考虑多个样本即二维数组的情况
    返回: numpy数组类型
    """
    # 如果x中的值过大会导致指数计算的结果过大发生溢出：np.exp(x)会出现inf，最终的结果会出现nan，可考虑减去每一个样本对应x中的最大值
    Todo()

def cross_entropy_error(y, t):
    # y是预测标签，t是真实标签
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原x
        it.iternext()   
        
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    df即dout,上一个节点传递过来的导数值
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        
        x[ix] = oldval + h
        pos = f(x).copy()

        x[ix] = oldval - h
        neg = f(x).copy()
        
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def sgd(params, grads, learning_rate = 0.001):
    for key in params.keys():
        params[key] -= learning_rate * grads[key] 


def conv_forward_naive(x, w, b, conv_param):
    """可以用np.lib.stride_tricks.sliding_window_view产生滑动窗口
       可以用np.pad填充
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_p, w, b, conv_param)  返回填充后的x便于反向传播直接使用
    """
    
    Todo()
    

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x_p, w, b, conv_param) as in conv_forward_naive, x_p是填充后的

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    
    # Helper function (warning: numpy 1.20+ is required)
    to_fields = np.lib.stride_tricks.sliding_window_view

    x_pad, w, b, conv_param = cache       # extract parameters from cache
    s = conv_param['stride']        # stride:  up = down
    p = conv_param['pad'] # padding: up = right = down = left
    
    N, _, HO, WO = dout.shape              # output dims
    FN, C, FH, FW = w.shape                # filter dims
    
    dout = np.insert(dout, [*range(1, HO)] * (s-1), 0, axis=2)         # "missing" rows
    dout = np.insert(dout, [*range(1, WO)] * (s-1), 0, axis=3)         # "missing" columns
    dout_pad = np.pad(dout, ((0,), (0,), (FH-1,), (FW-1,)), 'constant') # for full convolution

    x_fields = to_fields(x_pad, (N, C, dout.shape[2], dout.shape[3]))   # input local regions w.r.t. dout
    dout_fields = to_fields(dout_pad, (N, FN, FH, FW))                   # dout local regions w.r.t. filter 
    w_rot = np.rot90(w, 2, axes=(2, 3))                                 # rotated kernel (for convolution)

    db = np.einsum('ijkl->j', dout)                                                # sum over
    dw = np.einsum('ijkl,mnopiqkl->jqop', dout, x_fields)                          # correlate
    dx = np.einsum('ijkl,mnopqikl->qjop', w_rot, dout_fields)[..., p:-p, p:-p] # convolve

    return dx, dw, db


def conv_forward_fast(x, w, b, conv_param):
    
    N, C, H, W = x.shape 
    FN, C, FH, FW = w.shape 

    pad = conv_param['pad']
    stride = conv_param['stride']

    out_h = 1 + int((H + 2*pad - FH) / stride) 
    out_w = 1 + int((W + 2*pad - FW) / stride)

    col = im2col(x, FH, FW, stride, pad)  
    col_W = w.reshape(FN, -1).T                

    out = np.dot(col, col_W) + b          
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    cache = (x, w, b, conv_param) 
    return out, cache


def conv_backward_fast(dout, cache):
    x, w, b, conv_param = cache
    FN, C, FH, FW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    dout = dout.transpose(0,2,3,1).reshape(-1, FN)
    
    col = im2col(x, FH, FW, stride, pad)  # 图片转成矩阵,会填充重复值
    col_w = w.reshape(FN, -1).T                # 权重转矩阵，直接reshape
    db = np.sum(dout, axis=0)
    dw = np.dot(col.T, dout)
    dw = dw.transpose(1, 0).reshape(FN, C, FH, FW)

    dcol = np.dot(dout, col_w.T)
    dx = col2im(dcol, x.shape, FH, FW, stride, pad)

    return dx, dw, db

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return int((input_size + 2*pad - filter_size) / stride + 1)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """image to column图片数据重排为一个大矩阵

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    重排后的矩阵恢复为图片数据
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def download_CIFAR10(dataset_dir='./src/dataset'):
    url_base = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = 'cifar-10-python.tar.gz'
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return
    
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base, file_path)
    # import requests
    # requests.get(url_base)
    print("Done")

    import tarfile 
    with tarfile.open(file_name) as file:
        file.extractall(dataset_dir) 


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte
