import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

data = np.loadtxt('data/score_mlr03.txt', unpack=False, dtype='float32')

train_data_x = data[0:-5, 0:-1]
train_data_y = data[0:-5, [-1]]

test_data_x = data[-5:, 0:-1]
test_data_y = data[-5:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: train_data_x, Y: train_data_y})
    
    if i % 1000 == 0:
        print(i, "Cost: ", cost_val)

print("Predicts final score: \n", sess.run(hypothesis, feed_dict={X: test_data_x}))