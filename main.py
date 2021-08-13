#!/usr/bin/env python

from options.base_parser import BaseParser
import os
import tensorflow as tf
from backbones import backbone_helpers



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    flags = BaseParser().parse()
    # need to add argparse
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpus
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    
    
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        class_model = backbone_helpers.get_backbone_model(flags= flags)
        model = class_model(flags= flags, sess = sess)

        # build graph
        model.build_model()

        if flags.phase == 'train' or flags.phase == 'all':
            # launch the graph in a session
            model.train()
        elif flags.phase == 'test' or flags.phase == 'all':
            model.test()
        elif flags.phase == 'extract_features' or flags.phase == 'all':
            model.extract_features()
        elif flags.phase == 'saliency_maps' or flags.phase == 'all':
            model.saliency_maps()

    

