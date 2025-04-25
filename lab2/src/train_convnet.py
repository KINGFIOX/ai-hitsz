# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from src.util import *

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

Todo()