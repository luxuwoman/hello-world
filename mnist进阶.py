from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("G:/mnist/", one_hot = True)    
#导入下载的mnist数据
#One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数的索引之外，它都是零值，它被标记为1。

#print(mnist.train.images.shape, mnist.train.labels.shape)    #测试导入是否成功

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder('float', [None, 784])  #x为占位符，输入任意数量的mnist图像，每一张图展平成784维的向量
y_ = tf.placeholder('float', [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  #truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  #创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #strides卷积窗口滑动方式 

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])  #1个输入通道，32个输出通道
b_conv1 = bias_variable([32]) #32个输出通道的偏置

x_image = tf.reshape(x, [-1,28,28,1])  #处理输入的图片，变成需要的形式

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #激活函数ReLU，得到32个28*28的矩阵
h_pool1 = max_pool_2x2(h_conv1)  #池化得到32个14*14的矩阵

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  #得到64个7*7的矩阵

#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
