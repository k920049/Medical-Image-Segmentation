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

from tensorflow.contrib.layers.python.layers import xavier_initializer
from tensorflow.python.layers.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout

class UserModel(Tower):

    def _build_network(self):
        # parameters for the network
        conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3]]
        conv_strides = [[1, 1], [1, 1], [1, 1], [1, 1]]
        conv_filters = [32, 64, 128, 256]
        pool_strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        upconv_kernel_size = [[2, 2], [2, 2], [2, 2], [2, 2]]
        upconv_strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        upconv_filters = [256, 128, 64, 32]
        upconv_map_sizes = [[32, 32], [64, 64], [128, 128], [256, 256]]
        relu = tf.nn.relu
        xavier = xavier_initializer()
        num_layers = 2

        image = self.x
        image.set_shape([None, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        stack = []
        # convolutional layers
        with tf.variable_scope("conv", reuse=tf.AUTO_REUSE):
            for layer in range(len(conv_kernel_sizes)):
                for cell in range(num_layers):
                    image = Conv2D(filters=conv_filters[layer],
                                   kernel_size=conv_kernel_sizes[layer],
                                   strides=conv_strides[layer],
                                   padding='same',
                                   activation=relu,
                                   kernel_initializer=xavier)(image)
                stack.append(image)
                image = MaxPooling2D(pool_size=pool_strides[layer],
                                     strides=pool_strides[layer])(image)

        # steady stage
        with tf.variable_scope("steady_conv", reuse=tf.AUTO_REUSE):
            for cell in range(num_layers):
                image = Conv2D(filters=1024,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               activation=relu,
                               kernel_initializer=xavier)(image)
            image = Dropout(rate=0.1)(image)

        # upconvolutional layers
        with tf.variable_scope("upconv", reuse=tf.AUTO_REUSE):
            for layer in range(len(upconv_kernel_size)):
                prev_image = stack.pop()
                prev_image = tf.image.resize_bilinear(prev_image,
                                                      size=upconv_map_sizes[layer])
                image = Conv2DTranspose(filters=upconv_filters[layer],
                                        kernel_size=upconv_kernel_size[layer],
                                        strides=upconv_strides[layer],
                                        kernel_initializer=xavier)(image)
                image = tf.concat([image, prev_image], axis=3)

                for cell in range(num_layers):
                    image = Conv2D(filters=upconv_filters[layer],
                                   kernel_size=conv_kernel_sizes[layer],
                                   strides=conv_strides[layer],
                                   padding='same',
                                   activation=relu,
                                   kernel_initializer=xavier)(image)
        # getting class logit
        with tf.variable_scope("class", reuse=tf.AUTO_REUSE):
            logit = Conv2D(filters=2,
                           kernel_size=[1, 1],
                           strides=[1, 1],
                           padding='same',
                           kernel_initializer=xavier)(image)

        return logit

    @model_property
    def inference(self):
        logit = self._build_network()
        logit = tf.nn.softmax(logit)
        return logit

    @model_property
    def loss(self):
        # loads label data and model result
        label = self.y
        label.set_shape([None, 256, 256, 3])
        logit = self._build_network()
        # get the labels and preprocess
        y = tf.slice(label, begin=[0, 0, 0, 1], size=[-1, label.shape[1], label.shape[2], 1])
        d = tf.slice(label, begin=[0, 0, 0, 0], size=[-1, label.shape[1], label.shape[2], 1])
        y = tf.cast(y, dtype=tf.float32)
        d = tf.cast(d, dtype=tf.float32)
        # get the cross entropy term
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logit)
        w = 10.0
        sigma = 5.0
        d = 2.0 * d + 1
        d = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), d)
        energy = w * tf.exp(-(tf.square(d) / (2.0 * sigma ** 2)))
        loss = energy * entropy
        loss = tf.reduce_sum(loss)
        # adding regularization term
        scope = tf.get_variable_scope()
        variables = tf.trainable_variables(scope.name)
        return self.regularize(loss, variables, 0.05, 'l2')

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
        grad = [(tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0), v) for g, v in grad]
        return grad