import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

tf.set_random_seed(777)
learning_rate = 0.001

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 5]), name='weight1')
b1 = tf.Variable(tf.random_normal([5]), name='bias1')
#Layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)
Layer1 = tf.nn.relu(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([5, 5]), name='weight2')
b2 = tf.Variable(tf.random_normal([5]), name='bias2')
#Layer2 = tf.sigmoid(tf.matmul(Layer1, W2)+b2)
Layer2 = tf.nn.relu(tf.matmul(Layer1, W2)+b2)

W3 = tf.Variable(tf.random_normal([5, 5]), name='weight3')
b3 = tf.Variable(tf.random_normal([5]), name='bias3')
#Layer3 = tf.sigmoid(tf.matmul(Layer2, W3)+b3)
Layer3 = tf.nn.relu(tf.matmul(Layer2, W3)+b3)

W4 = tf.Variable(tf.random_normal([5, 5]), name='weight4')
b4 = tf.Variable(tf.random_normal([5]), name='bias4')
#Layer4 = tf.sigmoid(tf.matmul(Layer3, W4)+b4)
Layer4 = tf.nn.relu(tf.matmul(Layer3, W4)+b4)

W5 = tf.Variable(tf.random_normal([5, 5]), name='weight5')
b5 = tf.Variable(tf.random_normal([5]), name='bias5')
#Layer5 = tf.sigmoid(tf.matmul(Layer4, W5)+b5)
Layer5 = tf.nn.relu(tf.matmul(Layer4, W5)+b5)

W6 = tf.Variable(tf.random_normal([5, 5]), name='weight6')
b6 = tf.Variable(tf.random_normal([5]), name='bias6')
#Layer6 = tf.sigmoid(tf.matmul(Layer5, W6)+b6)
Layer6 = tf.nn.relu(tf.matmul(Layer5, W6)+b6)

W7 = tf.Variable(tf.random_normal([5, 5]), name='weight7')
b7 = tf.Variable(tf.random_normal([5]), name='bias7')
#Layer7 = tf.sigmoid(tf.matmul(Layer6, W7)+b7)
Layer7 = tf.nn.relu(tf.matmul(Layer6, W7)+b7)

W8 = tf.Variable(tf.random_normal([5, 5]), name='weight8')
b8 = tf.Variable(tf.random_normal([5]), name='bias8')
#Layer8 = tf.sigmoid(tf.matmul(Layer7, W8)+b8)
Layer8 = tf.nn.relu(tf.matmul(Layer7, W8)+b8)

W9 = tf.Variable(tf.random_normal([5, 1]), name='weight9')
b9 = tf.Variable(tf.random_normal([1]), name='bias9')
hypothesis = tf.sigmoid(tf.matmul(Layer8, W9) + b9)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(100001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 10000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)