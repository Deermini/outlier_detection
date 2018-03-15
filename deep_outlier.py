# !usr/bin/env python
# -*- coding:utf-8 -*-

from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 首先导入数据，看一下数据的形式
positive_data=np.load("data/train_positive.npy")
negative_data=np.load("data/train_negative.npy")
data=np.vstack((positive_data,negative_data))
trainX, validX = train_test_split(data, test_size=0.1, random_state=33)

positive_data=[];negative_data=[];data=[]

lr = 1e-4
input_size = 3
timestep_size = 3
hidden_size = 64
layer_num = 1
class_num =2
batch_size=64

X = tf.placeholder(tf.float32, [None,input_size,timestep_size])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

lstm_cell=rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,state_is_tuple=True)
lstm_cell=rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)
mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*1, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state=tf.nn.dynamic_rnn(cell=mlstm_cell,inputs=X,initial_state=None,dtype=tf.float32,time_major=False)

h_state = outputs[:, -1, :]
# h_state = state[-1][1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr,beta1=0.5,beta2=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs=80;label_index=9
    for epoch in range(epochs):
        ntrainbatchs=np.ceil(1.0*trainX.shape[0]/batch_size)
        for i in range(int(ntrainbatchs)-1):
            start=batch_size*i;end=(i+1)*batch_size
            if end>trainX.shape[0]:
                batch_x,batch_y=trainX[start:,:label_index],trainX[start:,label_index:]
            else:
                batch_x,batch_y=trainX[start:end,:label_index],trainX[start:end,label_index:]
            batch_x = batch_x.reshape([batch_size, input_size, timestep_size])
            batch_y=batch_y.reshape([batch_size,class_num])
            if (i + 1) % 20 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
                batch_validx=validX[:,:label_index].reshape([-1,input_size, timestep_size])
                batch_validy=validX[:,label_index:].reshape([-1,class_num])
                test_accuracy=sess.run(accuracy, feed_dict={
                    X: batch_validx, y: batch_validy, keep_prob: 1.0})
                print("Iter%d, step %3d, training accuracy %5g,test accuracy %5g" % (epoch,(i+1),train_accuracy,test_accuracy))
            sess.run(train_op, feed_dict={X: batch_x, y: batch_y, keep_prob: 0.6})

    # 计算测试数据的准确率
    batch_validx = validX[:, :label_index].reshape([-1, input_size, timestep_size])
    batch_validy = validX[:, label_index:].reshape([-1, class_num])
    print("test accuracy %g"% sess.run(accuracy, feed_dict={
        X:batch_validx , y: batch_validy, keep_prob: 1.0}))
    y_pre=sess.run(y_pre, feed_dict={X:batch_validx , y: batch_validy, keep_prob: 1.0})
    # for i in range(10):
    #     print("prediction:",y_pre[i])
    #     print("True:",batch_validy[i])
    #     print()
