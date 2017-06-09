
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb
from sklearn.utils import shuffle
from batch_generator import batch_gen
import matplotlib.pyplot as plt
# load mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels
pdb.set_trace()
X_train, y_train = shuffle(X_train, y_train)

def gen_noise(batch_size, dim):
	return np.random.normal(size=(batch_size, dim))

# Model definition
# Generator
z = tf.placeholder(tf.float32, [None, 100])
W_g1 = tf.get_variable('W_g1', shape=[100, 128], initializer=tf.contrib.layers.xavier_initializer())
W_g2 = tf.get_variable('W_g2', shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
b_g1 = tf.Variable(tf.zeros([128]))
b_g2 = tf.Variable(tf.zeros([784]))
gen_params = [W_g1, W_g2, b_g1, b_g2]
# Discriminator
x = tf.placeholder(tf.float32, [None, 784])
W_d1 = tf.get_variable('W_d1', shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
W_d2 = tf.get_variable('W_d2', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
b_d1 = tf.Variable(tf.zeros([128]))
b_d2 = tf.Variable(tf.zeros([1]))
dis_params = [W_d1, W_d2, b_d1, b_d2]
# Feed-forward eq of generator
g_hidden = tf.nn.relu(tf.add(tf.matmul(z, W_g1), b_g1))
g_z = tf.nn.relu(tf.add(tf.matmul(g_hidden, W_g2), b_g2))


# Feed-forward eq of discriminator
d_hidden = tf.nn.relu(tf.add(tf.matmul(x, W_d1), b_d1))
d_x = tf.nn.sigmoid(tf.add(tf.matmul(d_hidden, W_d2), b_d2))

d_g_h = tf.nn.relu(tf.add(tf.matmul(g_z, W_d1), b_d1))
d_g_z = tf.nn.sigmoid(tf.add(tf.matmul(d_g_h, W_d2), b_d2))


# Loss functions
J_d = tf.reduce_mean(tf.log(d_x) + tf.log(1 - d_g_z))
J_g = tf.reduce_mean(tf.log(d_g_z))

# Optimizers for generator and discriminator
d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(-1 * J_d, var_list=dis_params)
g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(-1 * J_g, var_list=gen_params)

# Initialize all variables
init = tf.global_variables_initializer()
# Simple feed forward for sanity-check
with tf.Session() as sess:
	sess.run(init)
	n_batches = int(X_train.shape[0]/float(64))
    #n_batches = int(math.ceil(n_batches))
	for i in xrange(1, 100000):
		print "---------__Training discriminator -------------"
		for k in range(1):
			try:
				x_batch = batch_gen(X_train, 32).next()
				if x_batch.shape[0] > 0:
					z_batch = gen_noise(x_batch.shape[0], 100)
					_, loss_batch_d = sess.run([d_optimizer, J_d], feed_dict={z:z_batch, x:x_batch})
					print "loss for batch %d for discriminator is %f"%(i, loss_batch_d)
			except Exception as e:
				print e
		z_batch = gen_noise(32, 100)
		_, loss_batch_g = sess.run([g_optimizer, J_g], feed_dict={z:z_batch})
		print "loss for batch %d for generator is %f"%(i, loss_batch_g)
		# Sampling from generator
		if i % 1000 == 0:
			z_batch = gen_noise(1, 100)
			img = sess.run([g_z], feed_dict={z:z_batch})
			plt.figure(figsize=(3, 3))
			plt.imshow(img[0].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
			plt.tight_layout()
			plt.show()
pdb.set_trace()
