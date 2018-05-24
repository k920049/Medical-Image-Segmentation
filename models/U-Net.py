import tensorflow as tf
import numpy as np

# import additional modules
import sys
sys.path.append("/Users/jeasungpark/Plugins/DIGITS/digits/tools/tensorflow")
from model import Tower
from utils import model_property
import utils as digits

from tensorflow.python import Tensor
from tensorflow.python.keras.utils import normalize

class UserModel(Tower):

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
        init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        num_layers = 2

        image = self.x
        image.set_shape([None, 256, 256, 3])
        stack = []
        # convolutional layers
        with tf.variable_scope("conv", reuse=tf.AUTO_REUSE):
            for cell in range(len(conv_kernel_sizes)):
                for i in range(num_layers):
                    image = tf.layers.conv2d(inputs=image,
                                             filters=conv_filters[cell],
                                             kernel_size=conv_kernel_sizes[cell],
                                             strides=conv_strides[cell],
                                             kernel_initializer=init,
                                             padding='same',
                                             activation=relu)
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
                                         kernel_initializer=init,
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
                                                   strides=upconv_strides[cell],
                                                   kernel_initializer=init)
                # concatenate two maps
                image = tf.concat([cropped_map, image], axis=3)
                # additional convolutional layers
                for i in range(num_layers):
                    image = tf.layers.conv2d(inputs=image,
                                             filters= upconv_filters[cell],
                                             kernel_size=conv_kernel_sizes[cell],
                                             strides=conv_strides[cell],
                                             kernel_initializer=init,
                                             padding='same',
                                             activation=relu)
        # final label
        logit = tf.layers.conv2d(inputs=image, filters=2, kernel_size=[1, 1], strides=[1, 1], padding='same')
        return logit

    @model_property
    def loss(self):
        # loads label data and model result
        label = self.y
        label.set_shape([None, 256, 256, 3])
        logit = self.inference
        class_prob = tf.nn.softmax(logit)
        # get the labels and preprocess
        y = tf.slice(label, begin=[0, 0, 0, 0], size=[-1, label.shape[1], label.shape[2], 2])
        d = tf.slice(label, begin=[0, 0, 0, 2], size=[-1, label.shape[1], label.shape[2], 1])
        y = tf.cast(y, dtype=tf.float32)
        d = tf.cast(d, dtype=tf.float32)
        # filters the result(2 classes)
        p = tf.multiply(class_prob, y)
        log_p = tf.log(p)
        log_p = tf.clip_by_value(log_p, clip_value_min=-10.0, clip_value_max=0.0)
        log_p = tf.reduce_mean(log_p, axis=3)
        # preparing hyperparameters
        sigma = 5.0
        w_0 = 10.0
        w_c_0 = 0.03 * tf.ones_like(d, dtype=tf.float32)
        w_c_1 = 32.3 * tf.ones_like(d, dtype=tf.float32)
        w_c = tf.concat([w_c_0, w_c_1], axis=3)
        w_c = tf.reduce_sum(w_c * y, axis=3)
        # assume we use l_1 distance, that is, d_2 = d_1 + 1
        d = 2.0 * d + 1
        mean, covariance = tf.nn.moments(d, axes=[0, 1, 2, 3])
        d = (d - mean) / tf.sqrt(covariance)
        energy = w_0 * tf.exp(-(tf.square(d) / (2.0 * sigma**2)))
        energy = tf.reduce_sum(energy, axis=3)
        weight = w_c + energy
        loss = tf.reduce_sum(weight * log_p)
        # regularization step
        scope = tf.get_variable_scope()
        params = tf.trainable_variables(scope.name)
        loss = self.regularize(-loss, params, 0.05, 'l2')

        return loss

    def regularize(self, cost, params, reg_val, reg_type):
        """
        Return regularized cost
        :param cost: cost to regularize
        :param params: list of parameters
        :param reg_val: multiplier for regularizer
        :param reg_type: accepted types of regularizer(options: 'l1' or 'l2'
        :param reg_spec:
        :return:
        """

        l1 = lambda p: tf.reduce_sum(tf.abs(p))
        l2 = lambda p: tf.reduce_sum(tf.square(p))
        rFxn = {}
        rFxn['l1'] = l1
        rFxn['l2'] = l2

        if reg_type == 'l2' or reg_type == 'l1':
            assert reg_val is not None, 'Expecting reg_val to be specified'
            regularizer = 0.0
            for p in params:
                regularizer = regularizer + rFxn[reg_type](p)
            return cost + reg_val * regularizer
        else:
            return cost

    def gradientUpdate(self, grad):
        grad = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in grad]
        return grad