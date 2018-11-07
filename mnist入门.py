from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("G:/mnist/", one_hot = True)    
#导入下载的mnist数据
#One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数的索引之外，它都是零值，它被标记为1。

#print(mnist.train.images.shape, mnist.train.labels.shape)    #测试导入是否成功

import tensorflow as tf
x = tf.placeholder('float', [None, 784])  #x为占位符，输入任意数量的mnist图像，每一张图展平成784维的向量

#一个Variable代表一个可修改的张量。
W = tf.Variable(tf.zeros([784, 10]))  #W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
b = tf.Variable(tf.zeros([10]))  #b的形状是[10]，所以我们可以直接把它加到输出上面。

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder('float', [None, 10])  #y_就是上面的y，是一个输入的固定的值
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #计算交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#定义了一个训练方法，在后面作为session.run()的参数

init = tf.initialize_all_variables() #定义一个初始化的函数动作，将前面定义的W和b初始化为零

sess = tf.Session() #定义一个session
sess.run(init) #启动这个session并初始化变量

for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)#随机抓取训练数据mnist中的100个批处理数据点
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})#用这些数据点作为参数替换之前的占位符来运行train_step


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #比较两个有y和y_对应的标签序列，返回一个bool类型的序列
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #cast()转换类型，reduce_mean()获得平均值
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))