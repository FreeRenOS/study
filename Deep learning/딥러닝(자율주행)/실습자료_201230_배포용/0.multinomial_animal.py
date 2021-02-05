import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

xy = np.loadtxt('data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, classes)  # N x 1 x 7
print("one_hot_encoding", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # N x 7
print("reshape: one_hot_encoding", Y_one_hot)

W = tf.Variable(tf.random_normal([16, classes]), name='weight')
b = tf.Variable(tf.random_normal([classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cross_entropy)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(opt, feed_dict={X: x_data, Y: y_data})
        if i % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                i, loss, acc))

    print("Training Done!!!")
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} Real Y: {}".format(p == int(y), p, int(y)))