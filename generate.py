import main
import os
import tensorflow as tf
import numpy as np
from scipy.misc import imsave as ims
from utils import merge

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('training/train-808.meta')
	# model = main.LatentAttention()
	# sess.run(tf.global_variables_initializer())
	saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + '/training'))

	op = sess.graph.get_operations()
	print([m.values() for m in op][1])

	for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		print(i)   # i.name if you want just a name

	print(sess.run('recognition/d_h1/w:0'))



	graph = tf.get_default_graph()





	dense_w = graph.get_tensor_by_name('generation/z_matrix/Matrix:0') # shape=(40, 4096) dtypeefloat32_ref>
	dense_b = graph.get_tensor_by_name('generation/z_matrix/bias:0') # shape=(4096,) dtype=eloat32_ref>
	conv1_w = graph.get_tensor_by_name('generation/g_h1/w:0') # shape=(5, 5, 512, 1024) dtype=loat32_ref>
	conv1_b = graph.get_tensor_by_name('generation/g_h1/b:0') # shape=(512,) dtype=float32_ref>
	conv2_w = graph.get_tensor_by_name('generation/g_h2/w:0') # shape=(5, 5, 256, 512) dtype=float32_ref>
	conv2_b = graph.get_tensor_by_name('generation/g_h2/b:0') # shape=(256,) dtype=float32_ref>
	conv3_w = graph.get_tensor_by_name('generation/g_h3/w:0') # shape=(5, 5, 128, 256) dtype=float32_ref>
	conv3_b = graph.get_tensor_by_name('generation/g_h3/b:0') # shape=(128,) dtype=float32_ref>
	conv4_w = graph.get_tensor_by_name('generation/g_h4/w:0') # shape=(5, 5, 64, 128) dtype=float32_ref>
	conv4_b = graph.get_tensor_by_name('generation/g_h4/b:0') # shape=(64,) dtype=float32_ref>
	conv5_w = graph.get_tensor_by_name('generation/g_h5/w:0') # shape=(5, 5, 32, 64) dtype=float32_ref>
	conv5_b = graph.get_tensor_by_name('generation/g_h5/b:0') # shape=(32,) dtype=float32_ref>
	conv6_w = graph.get_tensor_by_name('generation/g_h6/w:0') # shape=(5, 5, 16, 32) dtype=float32_ref>
	conv6_b = graph.get_tensor_by_name('generation/g_h6/b:0') # shape=(16,) dtype=float32_ref>
	conv7_w = graph.get_tensor_by_name('generation/g_h7/w:0') # shape=(5, 5, 3, 16) dtype=float32_ref>
	conv7_b = graph.get_tensor_by_name('generation/g_h7/b:0') # shape=(3,) dtype=float32_ref>


	def generate_from_z(z): # z is Nx40
		N = 10
		print("This many images being generated:", N)
		z_develop = tf.matmul(z, dense_w) + dense_b # N x 4096
		z_matrix = tf.nn.relu(tf.reshape(z_develop, [N, 2, 2, 1024]))

		h1 = tf.nn.relu(tf.nn.conv2d_transpose(z_matrix, conv1_w, output_shape=[N, 4, 4, 512], strides=[1,2,2,1]))
		h2 = tf.nn.relu(tf.nn.conv2d_transpose(h1, conv2_w, output_shape=[N, 8, 8, 256], strides=[1,2,2,1]))
		h3 = tf.nn.relu(tf.nn.conv2d_transpose(h2, conv3_w, output_shape=[N, 16, 16, 128], strides=[1,2,2,1]))
		h4 = tf.nn.relu(tf.nn.conv2d_transpose(h3, conv4_w, output_shape=[N, 32, 32, 64], strides=[1,2,2,1]))
		h5 = tf.nn.relu(tf.nn.conv2d_transpose(h4, conv5_w, output_shape=[N, 64, 64, 32], strides=[1,2,2,1]))
		h6 = tf.nn.relu(tf.nn.conv2d_transpose(h5, conv6_w, output_shape=[N, 128, 128, 16], strides=[1,2,2,1]))
		h7 = tf.nn.sigmoid(tf.nn.conv2d_transpose(h6, conv7_w, output_shape=[N, 256, 256, 3], strides=[1,2,2,1]))

		return h7



	# model.list_all_training_images('./data')

	# batch = model.get_training_images(10)
	# print("One image from batch:", batch[0])
	my_z = tf.truncated_normal([10,40], 0.0, 1.0)
	generated_images = generate_from_z(my_z)

	print(generated_images.get_shape().as_list())
	

	# generated_images = sess.run(model.generated_images, feed_dict={model.images: batch})
	# print(generated_images.shape)
	ims('my_generated_images.jpg', merge(generated_images.eval(), [5,2]))

	# print("One image:", generated_images[0])

