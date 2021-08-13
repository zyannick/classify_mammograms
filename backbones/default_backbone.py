

import time

import cv2

from ops import *
from utils import *
from data_helpers import *
import pickle
from tqdm import tqdm
import keras
from termcolor import colored
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
#from viz.visualization import vizualize_saliency

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

class Default_model(object):
    def __init__(self, sess, flags):
        self.model_name = flags.backbone
        self.flags = flags
        self.sess = sess
        self.dataset_name = flags.dataset_name

        self.checkpoint_path = flags.checkpoint_path
        self.log_dir = flags.log_dir

        self.image_size = None
        self.c_dim = 1
        self.label_dim = flags.num_classes


        self.n_epochs = flags.num_epochs
        self.batch_size = flags.batch_size

        self.init_lr = flags.init_lr
        self.current_epoch = 0

        self.nb_ebeddings = 216

        self.setup_datasets()


    def setup_datasets(self):
        self.init_step = 0
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.train_data_list, self.test_data_list = get_data(self.flags)

        self.images_shape = [None, self.image_size, self.image_size, self.c_dim]
        self.labels_shape = [None, self.label_dim]



    def network(self, x, is_training=True, reuse=False):
        raise NotImplementedError


    def build_model(self):
        """ Graph Input """
        self.train_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='train_inputs')
        self.test_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.features = tf.placeholder(tf.float32, [self.batch_size, 1], name='train_labels')

        self.current_lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_images)
        self.test_logits = self.network(self.test_images, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy, self.train_probabilities, self.train_prediction = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy, self.test_probabilities, self.test_prediction = classification_loss(logit=self.test_logits, label=self.test_labels)
        
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss
        
        self.score_max = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='scores_max')

        self.gradient = tf.gradients(self.test_logits[:,np.argmax(self.test_logits, -1)],self.test_images)

        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.current_lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])


    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        lr = self.flags.init_lr * (0.1 ** (self.current_epoch // self.flags.step_size))
        return lr


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        best_loss = np.inf
        best_epoch = 0

        # summary writer
        self.writer = tf.summary.FileWriter(os.path.join(self.checkpoint_path, self.log_dir, self.model_dir), self.sess.graph)


        # loop for epoch
        start_time = time.time()
        counter = 0
        batch_idxs = len(self.train_data_list) // self.flags.batch_size
        test_batch_idx = len(self.test_data_list) // self.flags.batch_size
        while self.current_epoch < self.flags.num_epochs:
            learning_rate = self.adjust_learning_rate()

            print('epoch {}'.format(self.current_epoch))

            tot_loss = 0.0
            train_preds = None
            train_labels = None
            train_probs = None

            test_preds = None
            test_labels = None
            test_probs = None
            # get batch data
            # get batch data
            for _, idx in tqdm(enumerate(np.arange(batch_idxs))):
                train_batch_images, train_batch_labels, _ = get_batch(self.flags.data_root, self.flags.batch_size, self.flags.dataset_name, self.train_data_list, self.image_size, self.flags.channel, self.flags.num_classes)

                train_feed_dict = {
                    self.train_images : train_batch_images,
                    self.train_labels : train_batch_labels,
                    self.current_lr : learning_rate
                }

                test_batch_images, test_batch_labels, _ = get_batch(self.flags.data_root, self.flags.batch_size, self.flags.dataset_name, self.test_data_list, self.image_size, self.flags.channel, self.flags.num_classes)


                test_feed_dict = {
                     self.test_images : test_batch_images,
                     self.test_labels : test_batch_labels
                 }

            
                # update network
                _, summary_str, train_loss, train_accuracy, probabilities, ground_truth, logits, predictions = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy, self.train_probabilities, self.train_labels, self.train_logits, self.train_prediction], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                tot_loss += train_loss

                if train_labels is None or train_preds is None:
                    train_preds = []
                    train_labels = []
                    train_probs = []

                train_probs.extend([p[1] for p in probabilities.tolist()])
                train_preds.extend(predictions.tolist())
                train_labels.extend(ground_truth.tolist())

                # test
                summary_str, test_loss,  test_prediction, test_probability, test_ground_truth = self.sess.run([self.test_summary, self.test_loss,  self.test_prediction, self.test_probabilities, self.test_labels], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                if test_labels is None or test_preds is None:
                    test_preds = []
                    test_labels = []
                    test_probs = []

                test_probs.extend([p[1] for p in test_probability.tolist()])
                test_preds.extend(test_prediction.tolist())
                test_labels.extend(test_ground_truth.tolist())

                # display training status
                counter += 1

            
            train_preds = np.array(train_preds)
            train_labels = np.array(train_labels)
            train_probs = np.array(train_probs)

            accuracy, f1_sco, cf, auc, soft_auc = compute_metrics(
                predictions=train_preds,
                labels=train_labels,
                probabilities=train_probs)

            affichage = 'Train results : ite:{}, Loss:{:.4f}, learning_rate: {}, accuracy:{:.4f}, f1score:{:.4f}, confusion_matrix:{}, auc:{:.4f}, soft_auc:{:.4f}'.format(self.current_epoch, tot_loss, learning_rate, accuracy, f1_sco,[cf[0, 0], cf[0, 1], cf[1, 0], cf[1, 1]], auc, soft_auc)
            if best_loss > tot_loss:
                c = 'green'
                best_loss = tot_loss
                best_epoch = self.current_epoch
            else:
                c = 'red'

            print(colored(affichage, c))

            test_preds = np.array(test_preds)
            test_labels = np.array(test_labels)
            test_probs = np.array(test_probs)

            test_accuracy, test_f1_sco, t_cf, test_auc, test_soft_auc = compute_metrics(
                predictions=test_preds,
                labels=test_labels,
                probabilities=test_probs)

            affichage_test =  'Test results : ite:{}, accuracy:{:.4f}, f1score:{:.4f}, confusion_matrix:{}, auc:{:.4f}, soft_auc:{:.4f}'.format(self.current_epoch, test_accuracy, test_f1_sco,[t_cf[0, 0], t_cf[0, 1], t_cf[1, 0], t_cf[1, 1]], test_auc, test_soft_auc)

            print(colored(affichage_test, 'blue'))

            #print('Epoch: [{:d}] [{:d}/{:d}] time: {:4.4f}, loss: {:.4f} train_accuracy: {:.4f}, test_accuracy: {:.4f}, learning_rate : {:.4f}'.format(self.current_epoch, idx, batch_idxs, time.time() - start_time, train_loss,  train_accuracy, test_accuracy, learning_rate))
            #print('\n\n')
            self.current_epoch += 1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_path, counter)

        # save model for final step
        self.save(self.checkpoint_path, counter)

    def extract_features(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        intermediary = os.path.join(self.checkpoint_path, self.model_dir, 'features')
        utils.mkdirs(intermediary)
        self.load(self.model_dir)

        batch_idxs = len(self.test_data_list)

        images_paths = []

        image_paths = []

        test_preds = None
        test_labels = None
        test_probs = None

        for _,idx in tqdm(enumerate(np.arange(batch_idxs))):

            single_image, single_label, single_path = get_batch(self.flags.data_root,  self.flags.batch_size,self.flags.dataset_name, self.train_data_list, self.image_size, self.flags.channel, self.flags.num_classes, idx)

            viz_feed_dict = {
                self.test_images : single_image,
                self.test_labels : single_label
            }

            features, test_prediction, test_probability, test_ground_truth = self.sess.run(
                [ self.features, self.test_prediction, self.test_probabilities, self.test_labels],
                feed_dict=viz_feed_dict)

            if test_labels is None or test_preds is None:
                test_preds = []
                test_labels = []
                test_probs = []


            test_probs.extend([p[1] for p in test_probability.tolist()])
            test_preds.extend(test_prediction.tolist())
            test_labels.extend(test_ground_truth.tolist())


            name = str(idx).zfill(8)

            images_paths.append(single_path[0])

            with open(os.path.join(intermediary, 'data_' + name + '.pkl'),"wb") as fout:
                pickle.dump(features, fout, protocol=pickle.HIGHEST_PROTOCOL)

            image_paths.append(single_path[0])

        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_probs = np.array(test_probs)

        if np.ndim(test_labels) == 2:
            test_labels = np.argmax(test_labels, axis=-1)

        if np.ndim(test_preds) == 2:
            test_preds = np.argmax(test_preds, axis=-1)

        utils.plot_tsne_features( os.path.join(self.checkpoint_path, self.model_dir) ,intermediary,  image_paths, test_preds, test_labels)


    def get_saliency_map(self, image, label,iters=1):

        # print(image.shape)

        test_feed_dict = {
                self.test_images: image,
                self.test_labels: label
            }

        logits = self.sess.run([self.test_logits], feed_dict=test_feed_dict)
        logits = np.asfarray(logits)

        grad_mtx = np.zeros((4,224,224))
        for i in tqdm(range(iters)): # single backprop as iters = 1, by default
            grad = self.sess.run([self.gradient], feed_dict={self.test_images:image, self.test_labels:label})
            grad = np.array(grad).squeeze()
            grad_mtx+=grad
        
        return grad_mtx

    def saliency_maps(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        saliency_dir = os.path.join(self.checkpoint_path, self.model_dir, 'saliency')
        utils.mkdirs(saliency_dir)

        self.load(self.model_dir)

        batch_idxs = len(self.test_data_list)

        print('epoch {}'.format(self.current_epoch))

        for idx in range(batch_idxs):
            test_batch_images, test_batch_labels , images_names = get_batch(self.flags.data_root,  self.flags.batch_size,self.flags.dataset_name, self.train_data_list, self.image_size, self.flags.channel, self.flags.num_classes, index= idx)

            saliency_map = self.get_saliency_map(test_batch_images, test_batch_labels)

            for i in range(saliency_map.shape[0]):

                slec = saliency_map[0]
                slec = np.squeeze(slec)

                slec -= slec.min()  # ensure the minimal value is 0.0
                if slec.max() != 0:
                    slec /= slec.max()  # maximum value in image is now 1.0

                slec = cv2.applyColorMap((slec * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                cv2.imwrite(os.path.join(saliency_dir, 'saliency_' + images_names[i].split(os.sep)[-1]), slec)

 
 
    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_path)

        print()

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise ValueError('No Model found')

        batch_idxs = len(self.test_data_list)

        test_preds = None
        test_labels = None
        test_probs = None

        for _, _ in tqdm(enumerate(np.arange(batch_idxs))):

            test_batch_images, test_batch_labels, _ = get_batch(self.flags.data_root, self.flags.batch_size,self.flags.dataset_name, self.train_data_list, self.image_size, self.flags.channel, self.flags.num_classes)

            test_feed_dict = {
                self.test_images: test_batch_images,
                self.test_labels: test_batch_labels
            }

            summary_str, test_loss,  test_prediction, test_probability, test_ground_truth = self.sess.run([self.test_summary, self.test_loss,  self.test_prediction, self.test_probabilities, self.test_labels], feed_dict=test_feed_dict)

            if test_labels is None or test_preds is None:
                test_preds = []
                test_labels = []
                test_probs = []

            test_probs.extend([p[1] for p in test_probability.tolist()])
            test_preds.extend(test_prediction.tolist())
            test_labels.extend(test_ground_truth.tolist())


        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_probs = np.array(test_probs)

        test_accuracy, test_f1_sco, t_cf, test_auc, test_soft_auc = compute_metrics(
                predictions=test_preds,
                labels=test_labels,
                probabilities=test_probs)

        affichage_test =  'Test results : ite:{}, accuracy:{:.4f}, f1score:{:.4f}, confusion_matrix:{}, auc:{:.4f}, soft_auc:{:.4f}'.format(self.current_epoch, test_accuracy, test_f1_sco,[t_cf[0, 0], t_cf[0, 1], t_cf[1, 0], t_cf[1, 1]], test_auc, test_soft_auc)

        print(colored(affichage_test, 'blue'))