# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import time
import numpy as np

from src.dataset.mnist import load_mnist
from src.two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01  
print('train size', train_size)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch = 5
iters_num = int(iter_per_epoch * epoch)
total_elapsed_time = 0
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 梯度计算
    start = time.time()
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    elapsed_time = time.time() - start
    total_elapsed_time += elapsed_time
    
    # 使用梯度下降法更新权重
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:  
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch {} train acc: {:.4f}, test acc: {:.4f}'.format(int(i / iter_per_epoch), train_acc,test_acc))  

print('gradient compute time: ', total_elapsed_time, 's')

# 训练后的模型输出test集前4张图片predict结果
sample_test = x_test[0:4]
sample_t_test = t_test[0:4]
sample_t_test = np.argmax(sample_t_test, axis=1)
sample_y = network.predict(sample_test)
sample_y = np.argmax(sample_y, axis=1)
print('predict label: ', sample_y, 'true label', sample_t_test)

import matplotlib.pyplot as plt 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(x_test[0].reshape(28,28), cmap ='gray')
ax2.imshow(x_test[1].reshape(28,28), cmap ='gray')
ax3.imshow(x_test[2].reshape(28,28), cmap ='gray')
ax4.imshow(x_test[3].reshape(28,28), cmap ='gray')
plt.show()