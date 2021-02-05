import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

data = np.loadtxt('data/sell_house.txt', unpack=False, dtype='float32')

x_test_data = data[-5:, 1:-1]
y_test_data = data[-5:, [-1]]
x_train = data[0:-5, 1:-1]
y_train = data[0:-5, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

opt = tf.train.GradientDescentOptimizer(learning_rate=2e-4)
train = opt.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200001):
    cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_train, Y: y_train})
    
    if i % 20000 == 0:
        print(i, "Cost: ", cost_val)

print("Predicts: \n", sess.run(hypothesis, feed_dict={X: x_test_data}))