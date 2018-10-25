import numpy as np
import tensorflow as tf

from collections import deque
from skimage import transform

STACK_SIZE = 4  # Eğitim için 4 frame'i üst üste koyuyoruz.


class Model():
    def __init__(self, action_size, state_size=[84, 84, 4], learning_rate=0.0002):
        # queue yapısını kullanıyoruz bu frame yığma işlemi için.
        # Yeni bir frame geldiği zaman en eski frame'i queue'dan çıkarır.

        self.stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)
        self.action_size = action_size

        self.graph = self.__create_model(state_size, learning_rate)
        self.__set_weights()

    def preprocess_frame(self, frame):
        # Crop the screen (remove the roof because it contains no information)
        cropped_frame = frame[30:-10, 30:-30]

        # Normalize Pixel Values
        normalized_frame = cropped_frame / 255.0

        # Resize
        preprocessed_frame = transform.resize(normalized_frame, [84, 84])

        return preprocessed_frame

    def stack_frames(self, state, is_new_episode):
        frame = self.preprocess_frame(state)

        if is_new_episode:
            self.stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)

            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)

            stacked_state = np.stack(self.stacked_frames, axis=2)

        else:
            self.stacked_frames.append(frame)

            stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state, self.stacked_frames

    def save_model(self):
        self.saver.save(self.sess, "./models/model.ckpt")


    # derin öğrenme modelimiz için yardımcı fonksiyonlar# derin
    def __create_variable(self, name, shape, init=None, std=None, wd=None):
        if init is None:
            if std is None:
                std = (2. / shape[0]) ** 0.5

            init = tf.truncated_normal_initializer(stddev=std)

        with tf.device('/cpu:0'):
            new_variables = tf.get_variable(name, shape, initializer=init, dtype=tf.float32)

        return new_variables

    def __dense_layer(self, name, inputs, units, activation=tf.nn.elu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer()):
        return tf.layers.dense(inputs, units,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               name=name)

    def __conv_layer(self, name, inputs, filters, kernel_size=[4, 4], strides=[2, 2],
                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()):
        return tf.layers.conv2d(inputs, filters, kernel_size,
                                strides=strides,
                                kernel_initializer=kernel_initializer,
                                name=name)

    def __batch_norm_layer(self, name, conv_layer, training=True, epsilon=1e-5):
        return tf.layers.batch_normalization(conv_layer,
                                             training=training,
                                             epsilon=epsilon,
                                             name=name)

    def __create_model(self, state_size, learning_rate):
        # derin öğrenme modelimiz
        dqn = tf.Graph()

        with dqn.as_default():
            with tf.name_scope('input'):
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
                self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
                self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            with tf.variable_scope('conv1') as scope:  # Giriş [84, 84, 4] --> Çıkış [20, 20, 32]
                conv = self.conv_layer('conv1', inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4])
                batch_norm = self.batch_norm_layer('batch_norm1', conv)
                out1 = tf.nn.elu(batch_norm, name="conv1_out")

            with tf.variable_scope('conv2') as scope:  # Giriş [20, 20, 32] --> çıkış[9, 9, 64]
                conv = self.conv_layer('conv2', out1, filters=64)
                batch_norm = self.batch_norm_layer('batch_norm2', conv)
                out2 = tf.nn.elu(batch_norm, name="conv2_out")

            with tf.variable_scope('conv3') as scope:  # Giriş [9, 9, 64] --> çıkış [3, 3, 128]
                conv = self.conv_layer('conv3', out2, filters=128)
                batch_norm = self.batch_norm_layer('batch_norm3', conv)
                out3 = tf.nn.elu(batch_norm, name="conv3_out")

            with tf.variable_scope('fully_connected1') as scope:  # Giriş [3, 3, 128] --> çıkış [1152]
                flatten = tf.layers.flatten(out3)
                fc1 = self.dense_layer('fc1', flatten, units=512)

            with tf.variable_scope('output') as scope:  # Giriş [1152] --> çıkış [3]
                self.output_ = self.dense_layer('output', fc1, units=3, activation=None)

            Q = tf.reduce_sum(tf.multiply(output_, self.actions_), axis=1)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            loss_ = tf.reduce_mean(tf.square(self.target_Q - Q))

            self.optimizer_ = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_)

            return dqn

    def __set_weights(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.sess = tf.Session()

            try:
                checkpoint_dir = tf.train.latest_checkpoint(checkpoint_dir="./models/")
                self.saver.restore(self.sess, save_path=checkpoint_dir)
                print("Restored checkpoint from:", checkpoint_dir)
            except:
                print("Initializing variables")
                self.sess.run(tf.global_variables_initializer())
