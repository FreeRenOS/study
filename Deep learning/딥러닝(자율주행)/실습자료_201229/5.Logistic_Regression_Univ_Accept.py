import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

data = np.loadtxt('data/Data_logistic.txt', unpack=False, dtype='float32')

train_data_x = data[0:-10, 0:-1]
train_data_y = data[0:-10, [-1]]

test_data_x = data[-10:, 0:-1]
test_data_y = data[-10:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W)+b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(20001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: train_data_x, Y: train_data_y})
        if i % 2000 == 0:
            print(i, cost_val)
    
    acc = sess.run(accuracy, feed_dict={X: test_data_x, Y: test_data_y})

print("Accuracy: ", acc)