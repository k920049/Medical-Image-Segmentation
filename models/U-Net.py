import tensorflow as tf
import numpy as np

# import additional modules
import sys
sys.path.append("/Users/jeasungpark/Plugins/DIGITS/digits/tools/tensorflow")
from model import Tower
from utils import model_property
import utils as digits

class U_Net(Tower):

    @model_property
    def inference(self):
        # parameters for the network
        conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3]]
        conv_strides = [[1, 1], [1, 1], [1, 1], [1, 1]]
        conv_filters = [64, 128, 256, 512]
        pool_strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        upconv_kernel_size = [[2, 2], [2, 2], [2, 2], [2, 2]]
        upconv_strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        upconv_filters = [512, 256, 128, 64]
        upconv_map_sizes = [[32, 32], [64, 64], [128, 128], [256, 256]]
        relu = tf.nn.relu
        num_layers = 2

        image = self.x
        stack = []
        # convolutional layers
        with tf.variable_scope("conv", reuse=tf.AUTO_REUSE):
            for cell in range(len(conv_kernel_sizes)):
                for i in range(num_layers):
                    image = tf.layers.conv2d(inputs=image,
                                             filters=conv_filters[cell],
                                             kernel_size=conv_kernel_sizes[cell],
                                             strides=conv_strides[cell],
                                             padding='same', activation=relu)
                stack.append(image)
                image = tf.layers.max_pooling2d(inputs=image,
                                                pool_size=pool_strides[cell],
                                                strides=pool_strides[cell])
        # steady stage
        with tf.variable_scope("steady_conv", reuse=tf.AUTO_REUSE):
            for i in range(num_layers):
                image = tf.layers.conv2d(inputs=image,
                                         filters=1024,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same')
        # upconvolutional layers
        with tf.variable_scope("upconv", reuse=tf.AUTO_REUSE):
            for cell in range(len(upconv_kernel_size)):
                # copy and crop
                copied_map = stack.pop()
                cropped_map = tf.image.resize_bilinear(images=copied_map, size=upconv_map_sizes[cell])
                # upconvolution
                image = tf.layers.conv2d_transpose(inputs=image,
                                                   filters=upconv_filters[cell],
                                                   kernel_size=upconv_kernel_size[cell],
                                                   strides=upconv_strides[cell])
                # concatenate two maps
                image = tf.concat([cropped_map, image], axis=0)
                # additional convolutional layers
                for i in range(num_layers):
                    image = tf.layers.conv2d(inputs=image,
                                             filters= upconv_filters[cell],
                                             kernel_size=conv_kernel_sizes[cell],
                                             strides=conv_strides[cell],
                                             padding='same',
                                             activation=relu)
        # final label
        logit = tf.layers.conv2d(inputs=image, filters=2, kernel_size=[1, 1], strides=[1, 1], padding='same')
        return logit

    @model_property
    def loss(self):
        # loads label data and model result
        label = self.y
        logit = self.inference
        class_prob = tf.nn.softmax(logit)
        # get the labels and preprocess
        n_batch = tf.shape(y)[0]
        y = tf.slice(y, begin=[0, 0, 0, 0], size=[n_batch, 2, y.shape[2], y.shape[3]])
        d = tf.slice(y, begin=[0, 2, 0, 0], size=[n_batch, 1, y.shape[2], y.shape[3]])
        y = tf.cast(y, dtype=tf.float32)
        d = tf.cast(d, dtype=tf.float32)
        y = tf.transpose(y, perm=[0, 2, 3, 1])
        d = tf.transpose(d, perm=[0, 2, 3, 1])
        # filters the result(2 classes)
        p = class_prob * y
        log_p = tf.log(p)
        # preparing hyperparameters
        sigma = 5.0
        w_0 = 10.0
        w_c_0 = 0.03 * tf.ones(shape=[n_batch, 1, y.shape[2], y.shape[3]], dtype=tf.float32)
        w_c_1 = 32.3 * tf.ones(shape=[n_batch, 1, y.shape[2], y.shape[3]], dtype=tf.float32)
        w_c = tf.concat([w_c_0, w_c_1], axis=0)
        # assume we use l_1 distance, that is, d_2 = d_1 + 1
        d = 2.0 * d + 1
        energy = w_0 * tf.exp(-(tf.square(d) / (2.0 * sigma**2)))
        weight = w_c + energy
        loss = tf.reduce_sum(weight * log_p)

        return loss