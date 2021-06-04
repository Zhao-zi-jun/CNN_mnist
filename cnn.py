import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 防止报错warning

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape):
    # 给一个shape,返回shape大小的初始化的随机值
    # tf.truncated_normal从截断的正态分布中输出随机值.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)   # 注意：tensorflow的变量必须定义为tf.Variable类型，只有定义成该类型才会根据后面反向传播进行改变


def bias_variable(shape):
    # 偏置值，初始是0.1，在训练过程中会优化
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # 卷积核移动步长为1,填充padding类型为SAME,可以不丢弃任何像素点, VALID丢弃边缘像素点
    # 计算给定的4-D input和filter张量的2-D卷积
    # input shape [batch, in_height, in_width, in_channels]
    # filter shape [filter_height, filter_width, in_channels, out_channels]
    # stride对应在这四维上的步长，默认[1,x,y,1]。控制卷积核的移动步数，其中第一个1和最后一个1是固定值，需要改变的是中间的两个数，即在x轴和y轴的移动步长。


def max_pool_2x2(x):
    # 采用最大池化，也就是取窗口中的最大值作为结果
    # x 是一个4维张量，shape为[batch,height,width,channels]
    # ksize表示pool窗口大小为2x2,也就是高2，宽2
    # strides，表示在height和width维度上的步长都为2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 模型训练后，计算测试集准确率
def compute_accuracy(v_xs, v_ys):   # v_xs是测试集的数据[10000,784],v_ys是测试集的标签[10000,10]
    global prediction
    # y_pre是将v_xs(test)输入模型后得到的预测值 (10000,10)
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})  # 把测试集的图片喂入训练好的模型，来做预测
    # argmax(axis) axis = 1 返回结果为：数组中每一行最大值所在“列”索引值
    # tf.equal返回布尔值，correct_prediction (10000，1)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 计算准确率
    # tf.cast将bool转成float32, tf.reduce_mean求均值，作为accuracy值(0到1)
    # 把布尔值correct_prediction，转换为浮点数[10000,1],true=1,false=0 ;在求均值，均值越高预测的越准确
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 创建两个占位符（暂时不进行赋值的元素叫占位符，run需要它们时得赋值），xs为输入网络的图像，ys为输入网络的图像标签
xs = tf.placeholder(tf.float32, [None, 784], name='x_input') # 行数即样本数，不固定，列数固定，每个样本必须784维
ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# 输入xs(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]，第四维是通道数1
# -1表示根据输入的样本的数量自动计算。
keep_prob = tf.placeholder(tf.float32)  # dropout参数
keep_prob_rate = 0.5
max_epoch = 1000  # 最大优化次数    注意：试验阶段可以把 max_epoch 的值调的小一点

'''
输入图片大小 W×W = 28*28
Filter大小 F×F = 5*5
步长 S = 1
padding的像素数 P=2 (当padding='SAME'，P=(F-1)/2 ) 

卷积后输出图片的大小计算公式：N = [(W − F + 2P )/S] +1   ， 即 N=[（28-5+2*2）/1 ]+1 =28
简单地计算公式，当padding='SAME'是，直接计算 N=W/S后向上取整 ,即 N=28/1=28
'''


# 卷积层1代码： 输入是28*28*1，卷积后输出是28*28*32，池化输出是14*14*32（图片的大小是28*28,通道是1，卷积核个数 filter= 32）
# conv1 layer 含pool ##
W_conv1 = weight_variable([5, 5, 1, 32])
# 初始化W_conv1为[5,5,1,32]的张量tensor，表示卷积核大小为5*5，1表示图像通道数（输入），32表示卷积核个数即输出32个特征图。
b_conv1 = bias_variable([32])
# 偏置项，参与conv2d中的加法，维度会自动扩展到28x28x32（广播）
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)   # 激活函数是relu
# output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32 卷积操作使用padding保持维度不变，只靠pool降维

# 卷积层2代码：输入是14*14*32，卷积后输出是14x14x64，池化后输出是7x7x64
# conv2 layer 含pool##
W_conv2 = weight_variable([5, 5, 32, 64])  # 同conv1，不过卷积核数增为64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)
# output size 7x7x64

# 把卷积层2的输出进行摊平操作，即reshape成[batch, 7*7*64]的张量，方便全连接层处理，摊平后的大小是 1*3136的向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])


# 全连接层1代码：输入是1*3136，输出,1*1024
# fc1 layer ##
# 含1024个神经元，初始化（3136，1024）的tensor    注意：全连接层1024神经元的个数是自己设置的
W_fc1 = weight_variable([7 * 7 * 64, 1024])         # w权重
b_fc1 = bias_variable([1024])                       #偏置项
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # tf.matmul就是计算矩阵相乘（叉乘）
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  #dropout防止过拟合


# 全连接层2代码：输入是1*1024，输出是1*10
# fc2 layer 含softmax层##
# 含10个神经元，初始化（1024，10）的tensor
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1
# 使用交叉熵函数计算loss值。
# reduce_mean函数是计算张量的(各个维度上)元素的平均值，reduction_indices=1表示取每一列的平均值。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))


# 使用ADAM优化器来做梯度下降 ：用来更新权重 ，学习率learning_rate=0.0001
learning_rate = 1e-4
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) # 目的是最小化交叉熵，通过更新权重与偏置项


# 上述阶段就是构建阶段，现在进入执行阶段，反复执行图中的训练操作，首先需要创建一个Session对象
# Session对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作
with tf.Session() as sess:
    # 初始化图中所有Variables(即权重和偏置项)
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化session

    # 总迭代次数(batch)为max_epoch=1000,每次取100张图做batch梯度下降。（测试集的样本总数是10000）注意优化次数是max_epoch
    for i in range(max_epoch):
        # mnist.train.next_batch 默认shuffle=True，随机读取，batch大小为100
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 此batch是个2维tuple，batch[0]是(100，784)的样本数据数组，batch[1]是(100，10)的样本标签数组，分别赋值给batch_xs, batch_ys
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: keep_prob_rate})  # 使用优化器做反向传播，来优化权重和偏置。
        # 暂时不进行赋值的元素叫占位符（如xs、ys），run需要它们时得赋值，feed_dict就是用来赋值的，格式为字典型
        # 每优化50次输出当前模型的好坏程度
        if (i+1) % 50 == 0:
            print("step %d, test accuracy %g" % (i+1, compute_accuracy(mnist.test.images, mnist.test.labels)))



