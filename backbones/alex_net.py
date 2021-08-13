

import time
from ops import *
from utils import *
from data_helpers import *
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import conv2d
from .default_backbone import Default_model

class Alexnet(Default_model):
    def __init__(self, sess, flags):
        super(Alexnet, self).__init__(sess, flags)
        self.image_size = 256
        self.setup_datasets()


    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
                        # 1st
            conv1 = conv2d(x, num_outputs=96,
                        kernel_size=[11,11], stride=4, padding="VALID",
                        activation_fn=tf.nn.relu)
            lrn1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001,beta=0.75)
            pool1 = max_pool2d(lrn1, kernel_size=[3,3], stride=2)

            # 2nd
            conv2 = conv2d(pool1, num_outputs=256,
                        kernel_size=[5,5], stride=1, padding="VALID",
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=tf.nn.relu)
            lrn2 = tf.nn.local_response_normalization(conv2, bias=2, alpha=0.0001, beta=0.75)
            pool2 = max_pool2d(lrn2, kernel_size=[3,3], stride=2)

            #3rd
            conv3 = conv2d(pool2, num_outputs=384,
                        kernel_size=[3,3], stride=1, padding="VALID",
                        activation_fn=tf.nn.relu)

            #4th
            conv4 = conv2d(conv3, num_outputs=384,
                        kernel_size=[3,3], stride=1, padding="VALID",
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=tf.nn.relu)

            #5th
            conv5 = conv2d(conv4, num_outputs=256,
                        kernel_size=[3,3], stride=1, padding="VALID",
                        biases_initializer=tf.ones_initializer(),
                        activation_fn=tf.nn.relu)
            pool5 = max_pool2d(conv5, kernel_size=[3,3], stride=2)

            #6th
            flat = flatten(pool5)
            fcl1 = fully_connected(flat, num_outputs=4096,
                                    biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
            dr1 = tf.layers.dropout(inputs=fcl1, rate=0.5, training=is_training)

            #7th
            fcl2 = fully_connected(dr1, num_outputs=4096,
                                    biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
            dr2 = tf.layers.dropout(inputs=fcl2, rate=0.5, training=is_training)
            self.features = dr2

            #output
            out = fully_connected(dr2, num_outputs=self.label_dim, activation_fn=None)
            return out


