####可视化train数据集图片
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_img = mnist.train.images
train_label = mnist.train.labels

for i in range(5):
    img = np.reshape(train_img[i, :], (28, 28))
    label = np.argmax(train_label[i, :])     #转换为阿拉伯数字
    plt.matshow(img, cmap='gray')   #灰度矩阵，0表示黑色，1表示白色
    plt.title('第%d张图片 标签为%d' %(i+1,label))
    plt.show()
