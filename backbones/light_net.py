

import time
from ops import *
from utils import *
from data_helpers import *
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import conv2d
from .default_backbone import Default_model

class Light_net(Default_model):
    def __init__(self, sess, flags):
        super(Light_net, self).__init__(sess, flags)
        self.image_size = 256
        self.setup_datasets()



    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
                        # 1st
            conv1 = conv2d(x, num_outputs=8,
                        kernel_size=[11,11], stride=4, padding="VALID",
                        activation_fn=tf.nn.relu)
            lrn1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001,beta=0.75)
            pool1 = max_pool2d(lrn1, kernel_size=[3,3], stride=2)

            #6th
            flat = flatten(pool1)
            fcl1 = fully_connected(flat, num_outputs=64,
                                    biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
            dr1 = tf.layers.dropout(inputs=fcl1, rate=0.5, training=is_training)

            #7th
            fcl2 = fully_connected(dr1, num_outputs=64,
                                    biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
            dr2 = tf.layers.dropout(inputs=fcl2, rate=0.5, training=is_training)

            self.features = dr2
            #output
            self.out = fully_connected(dr2, num_outputs=self.label_dim, activation_fn=None)
            return self.out


    def get_saliency_map(self,model, image, class_idx):
        with tf.GradientTape() as tape:
            tape.watch(image)
            predictions = model(image)
            
            loss = predictions[:, class_idx]
        
        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, image)
        
        # take maximum across channels
        gradient = tf.reduce_max(gradient, axis=-1)
        
        # convert to numpy
        gradient = gradient.numpy()
        
        # normaliz between 0 and 1
        min_val, max_val = np.min(gradient), np.max(gradient)
        smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())
        
        return smap



