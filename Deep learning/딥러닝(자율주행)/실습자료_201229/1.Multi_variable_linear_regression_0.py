import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x1 = [73, 93, 89, 96, 73]
x2 = [80, 88, 91, 98, 66]
x3 = [75, 93, 90, 100, 70]
y = [152, 185, 180, 196, 142]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X1 * w1 + X2 *w2 + X3 * w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    cost_val, hypothesis_val, _ = sess.run([cost, hypothesis, train], feed_dict={X1: x1, X2: x2, X3: x3, Y: y})
    
    if i % 50 == 0:
        print(i, "Cost: ", cost_val, "\nHypothesis: \n", hypothesis_val)