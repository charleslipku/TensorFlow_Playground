import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets(train_dir='MNIST_data/', one_hot=True)

# define variabels and placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
label = tf.placeholder(dtype=tf.float32,shape=[None,10])

# define the model, loss function and optimization method
y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cross_entropy)

# start training
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, label:batch_ys})

# evaluate the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))