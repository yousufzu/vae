import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.misc import imsave as ims
from PIL import Image
from utils import *
from ops import *

class LatentAttention():
    def __init__(self):
        self.n_samples = 0 #set later

        self.n_hidden = 500
        self.n_z = 40
        self.batchsize = 10
        self.so_far = 0 #how far we've gotten in the set
        self.my_images = [] # later set to self.list_all_training_images('./data')

        self.images = tf.placeholder(tf.float32, [None, 256*256*3])
        image_matrix = tf.reshape(self.images,[-1, 256, 256, 3])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z], 0, 1, dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(self.guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 256*256*3])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-6 + generated_flat) + (1-self.images) * tf.log(1e-6 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-6 + tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)



    def list_all_training_images(self, im_dir):
        # need to open the directory, store all the paths in self.images
        imlist = os.listdir(im_dir)
        ret = []
        for impath in imlist:
            print("Got image:", impath)
            ret.append(im_dir + "/" + impath)
        self.n_samples = len(ret)
        return ret

    def get_training_images(self, num=10):
        start = self.so_far
        end = start + num
        self.so_far = end

        if end > len(self.my_images): #wrap around
            start = 0
            end = start + num
            self.so_far = end

        image_subset = self.my_images[start:end]
        image_bytes = np.zeros((num, 3*256*256)) #TODO: change 1024

        i = 0
        for image in image_subset:
            im_jpg = Image.open(image)
            im_jpg.load()
            im_array = np.asarray(im_jpg).flatten() #Type is uint8!!!
            while im_array.shape[0] < 10: # fails to load sometimes
                im_array = np.asarray(im_array[0]).flatten()
            image_bytes[i] = (im_array - 128) / 128  # feature scaling
            i = i+1

        return image_bytes

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 3, 16, "d_h1")) # 256x256x3 -> 128x128x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 128x128x16 -> 64x64x32
            h3 = lrelu(conv2d(h2, 32, 64, "d_h3")) # 64x64x32 -> 32x32x64
            h4 = lrelu(conv2d(h3, 64, 128, "d_h4")) # 32x32x64 -> 16x16x128
            h5 = lrelu(conv2d(h4, 128, 256, "d_h5")) # 16x16x128 -> 8x8x256
            h6 = lrelu(conv2d(h5, 256, 512, "d_h6")) #8x8x256 -> 4x4x512
            h7 = lrelu(conv2d(h6, 512, 1024, "d_h7")) #4x4x512 -> 2x2x1024
            h7_flat = tf.reshape(h7,[self.batchsize, 2*2*1024])
            
            w_mean = dense(h7_flat, 2*2*1024, self.n_z, "w_mean")
            w_stddev = dense(h7_flat, 2*2*1024, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 2*2*1024, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 2, 2, 1024]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 4, 4, 512], "g_h1"))
            h2 = tf.nn.relu(conv_transpose(h1, [self.batchsize, 8, 8, 256], "g_h2"))
            h3 = tf.nn.relu(conv_transpose(h2, [self.batchsize, 16, 16, 128], "g_h3"))
            h4 = tf.nn.relu(conv_transpose(h3, [self.batchsize, 32, 32, 64], "g_h4"))
            h5 = tf.nn.relu(conv_transpose(h4, [self.batchsize, 64, 64, 32], "g_h5"))
            h6 = tf.nn.relu(conv_transpose(h5, [self.batchsize, 128, 128, 16], "g_h6"))
            h7 = tf.nn.sigmoid(conv_transpose(h6, [self.batchsize, 256, 256, 3], "g_h7"))

        return h7

    def train(self):
        self.my_images = self.list_all_training_images('./data')
        visualization = self.get_training_images(self.batchsize)
        reshaped_vis = visualization.reshape(self.batchsize, 256, 256, 3)
        reshaped_vis = (reshaped_vis * 127.5) + 127.5
        reshaped_vis = reshaped_vis.astype(np.uint8)
        ims("results/base.jpg", merge(reshaped_vis, [5,2]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(1000):
                for idx in range(int(self.n_samples / self.batchsize)):
                   
                    batch = self.get_training_images(self.batchsize)
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    
                    
                    
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize, 256, 256, 3)
                        generated_test = (generated_test * 127.5) + 127.5
                        generated_test = generated_test.astype(np.uint8)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test, [5,2]))
                        
                if epoch % 11 == 0:
                    saver.save(sess, os.getcwd()+"/training/train", global_step=epoch)

if __name__ == '__main__':
	model = LatentAttention()
	model.train()
