# coding: utf-8
import sys, os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from src.util import *

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 假设你已经加载了MNIST数据
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 初始化网络
input_dim = (1, 28, 28)
conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1}
hidden_size = 100
output_size = 10
network = SimpleConvNet(input_dim, conv_param, hidden_size, output_size)

# 训练网络
batch_size = 100
iters_num = 10000
train_size = x_train.shape[0]
max_iter = int(train_size / batch_size)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    grads = network.gradient(x_batch, t_batch)
    
    # 更新权重
    for key in ('W1', 'W2', 'W3'):
        network.params[key] -= 0.1 * grads[key]
    
    # 每100次打印一次损失和准确率
    if i % 100 == 0:
        loss = network.loss(x_batch, t_batch)
        print(f"Iteration {i}, Loss: {loss}")

# 测试准确率
accuracy = network.accuracy(x_test, t_test)
print(f"Test Accuracy: {accuracy}")
