import tensorflow as tf
import numpy as np
import utils as utils
import re
import datareader as dr
import os

from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.contrib import rnn

FLAGS = tf.flags.FLAGS

file_dir = "/home/sbie/github/VAD_KJT/Data/data_0302_2017/Aurora2withSE"
input_dir = file_dir + "/Noisy_Aurora_STFT_npy"
output_dir = file_dir + "/label_checked"

valid_file_dir = "/home/sbie/github/VAD_KJT/Data/data_0308_2017/Aurora2withNX"
valid_input_dir = valid_file_dir + "/Noisy_Aurora_STFT_npy/Babble/SNR_10"
valid_output_dir = valid_file_dir + "/labels"

logs_dir = "/home/sbie/github/VAD_KJT/logs"

reset = True  # remove all existed logs and initialize log directories
eval_only = False  # if True, skip the training phase
device = '/gpu:0'
if reset:
    os.popen('rm -rf ' + logs_dir + '/*')
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/valid')

num_valid_batches = 100
learning_rate = 0.001
num_steps = 32
batch_size = 32  # batch_size = 32
num_sbs = 16  # number of sub-bands
fft_size = 256
channel = 1
cell_size = 256
cell_out_size = cell_size
num_h1_sb_net = 128
num_h2_sb_net = 256
SMALL_NUM = 1e-4
max_epoch = int(1e6)
dropout_rate = 0.85
decay = 0.9  # batch normalization decay factor

assert fft_size % num_sbs == 0, "fft_size must be divisible by num_sbs."


def sb_sensor(bp, STFT):
    """
    get selected sub-bands
    :param bp: bernoulli process. shape = (batch_size, num_sbs, 1), the last dimension is for tiling.
    :param STFT: STFT. shape = (batch_size, fft_size/2, channel)
    :return: selected sub-bands. shape = (batch_size, fft_size/2, 1, channel)
    """
    bp = tf.tile(bp, (1, 1, tf.to_int32(fft_size / num_sbs)))
    bp = tf.reshape(bp, (batch_size, -1, 1))
    bp = tf.tile(bp, (1, 1, channel))
    selected_sbs = bp * STFT  # element-wise multiplication
    # selected_sbs = STFT
    selected_sbs = tf.expand_dims(selected_sbs, 2)  # expand dimension for convolution

    return selected_sbs


def sb_net(bp, STFT, reuse=False, is_training=True):
    """
    get sub-band network output (input to recurrent neural network)
    :param bp: bernoulli process. shape = (batch_size, num_sbs, 1), elements in bp are 0 or 1.
    :param STFT: STFT. shape = (batch_size, fft_size/2, channel)
    :param reuse: reuse factor for variable scope. shape = True or False
    :param is_training
    :return: sub-band network output. shape = (batch_size, fft_size/2, 1, channel)
    """
    selected_sbs = sb_sensor(bp, STFT)
    with tf.variable_scope("conv_net", reuse=reuse):
        conv_out = conv_net(selected_sbs)
        last_conv = tf.squeeze(utils.conv2lstm_layer(conv_out["max_pool4_3"], num_h1_sb_net))
    with tf.variable_scope("fc_net", reuse=reuse):
        last_fc = utils.batch_norm_affine_transform(tf.squeeze(bp), num_h1_sb_net, decay=decay,
                                                    name="sb_net_1", is_training=is_training)
        last_fc = tf.nn.relu(last_fc)
        # last_fc = tf.nn.relu(affine_transform(tf.squeeze(bp), num_h1_sb_net, name="sb_net_1"))  # TODO batch norm
    with tf.variable_scope("sb_net_out", reuse=reuse):

        last_conv = utils.batch_norm_affine_transform(last_conv, num_h2_sb_net, decay=decay,
                                                      name="sb_net_2", is_training=is_training)
        last_conv = tf.nn.relu(last_conv)
        last_fc = utils.batch_norm_affine_transform(last_fc, num_h2_sb_net, decay=decay,
                                                    name="sb_net_3", is_training=is_training)
        last_fc = tf.nn.relu(last_fc)
        sb_net_out = last_conv + last_fc
        # sb_net_out = tf.nn.relu(affine_transform(last_conv, num_h2_sb_net, name="sb_net_2")  # TODO batch norm
        #                         + affine_transform(last_fc, num_h2_sb_net, name="sb_net_3"))

    return sb_net_out


def conv_net(inputs, is_training=True):
    layers = (
        'index:1_1, type:conv, size:2, stride:1, fm:'+str(channel)+'->32',
        'index:1_2, type:relu',
        'index:1_3, type:max_pool',
        'index:2_1, type:conv, size:2, stride:1, fm:32->64',
        'index:2_2, type:relu',
        'index:2_3, type:max_pool',
        'index:3_1, type:conv, size:2, stride:1, fm:64->128',
        'index:3_2, type:relu',
        'index:3_3, type:max_pool',
        'index:4_1, type:conv, size:2, stride:1, fm:128->128',
        'index:4_2, type:relu',
        'index:4_3, type:max_pool'
        )

    net = {}
    current = inputs
    for i, name in enumerate(layers):
        spec = re.split(':|, |->', name)
        index = spec[1]
        kind = spec[3]
        if kind == 'conv':
            conv_shape, stride = utils.get_1d_conv_shape(name)
            kernels = utils.weight_variable(conv_shape, name=kind+index+"_w")
            bias = utils.bias_variable([conv_shape[-1]], name=kind+index+"_b")
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]

            current = utils.conv2d_basic(current, kernels, bias, stride=stride)
            current = tf.contrib.layers.batch_norm(current, decay=decay, is_training=is_training,
                                                   updates_collections=None)
        
        elif kind == 'relu':
            current = tf.nn.relu(current, name=kind+index)
        elif kind == 'max_pool':
            current = utils.max_pool_2x1(current)
        elif kind == 'avg_pool':
            current = utils.avg_pool_2x2(current)
        net[kind+index] = current

    return net


def affine_transform(x, output_dim, name=None):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """

    w = tf.get_variable(name+"_w", [x.get_shape()[1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+"_b", [output_dim], initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, w) + b


def get_sbs(cell_output, reuse=False, is_training=True):

    with tf.variable_scope("br_net", reuse=reuse):

        br_out = utils.batch_norm_affine_transform(cell_output, num_sbs, decay=decay,
                                                   name="br_net", is_training=is_training)
        br_out = tf.sigmoid(br_out)
        # br_out = tf.sigmoid(affine_transform(cell_output, num_sbs, name="br_net"))
    rand_seq = tf.random_uniform(br_out.get_shape().as_list(), minval=0, maxval=1)
    selected_sbs = tf.cast(tf.greater(br_out, rand_seq), tf.float32)
    return selected_sbs, br_out


def inference(inputs, keep_prob, is_training=True):
    """
    VAD based on recurrent method
    :param inputs: input list. length = num_steps, shape = (batch_size, fft_size, channel)
    :param keep_prob:
    :param is_training
    :return: cell_output, selected_sbs, br_out are list with length num_steps, shape = (batch_size, cell_out_size),
     (batch_size, num_sbs), (batch_size, num_sbs)
    """
    # initialization

    lstm_cell = rnn.LayerNormBasicLSTMCell(cell_size, dropout_keep_prob=keep_prob)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)

    cell_output_list = []
    selected_sbs_list = []
    br_out_list = []

    with tf.variable_scope("core_network"):
        for time_step in range(num_steps):
            if time_step is 0:
                reuse = False
                sbs_ini = tf.ones([batch_size, num_sbs, 1])
                sb_net_out = sb_net(sbs_ini, inputs[time_step], reuse=reuse, is_training=is_training)  # TODO batch norm
                (cell_output, cell_state) = lstm_cell(sb_net_out, initial_state)
                selected_sbs, br_out = get_sbs(cell_output, reuse=reuse, is_training=is_training)  # TODO batch norm
                selected_sbs = tf.expand_dims(selected_sbs, 2)

                cell_output_list.append(cell_output)
                selected_sbs_list.append(selected_sbs)
                br_out_list.append(br_out)
            else:
                reuse = True
                sb_net_out = sb_net(selected_sbs, inputs[time_step], reuse=reuse)
                tf.get_variable_scope().reuse_variables()  # for parameter sharing in lstm cell
                (cell_output, cell_state) = lstm_cell(sb_net_out, cell_state)
                selected_sbs, br_out = get_sbs(cell_output, reuse=reuse)
                selected_sbs = tf.expand_dims(selected_sbs, 2)

                cell_output_list.append(cell_output)
                selected_sbs_list.append(tf.squeeze(selected_sbs))

                br_out_list.append(br_out)

    # cell_output = tf.stack(cell_output_list, 1) , unused
    # selected_sbs = tf.stack(selected_sbs_list, 1) , unused
    # br_out = tf.stack(br_out_list, 1) , unused

    return cell_output_list, selected_sbs_list, br_out_list


def bernoulli_pmf(mean, sample):
    """
    calculate the probability of bernoulli process
    :param mean: mean. shape = (batch_size, num_sbs)
    :param sample: sample. shape = (batch_size, num_sbs)
    :return: p_br: shape = (batch_size, num_sbs)
    """

    p_br = sample * mean + (1 - sample) * (1 - mean)
    return p_br


def train(loss_val, var_list):
    initLr = 5e-3
    lrDecayRate = .99
    lrDecayFreq = 200
    momentumValue = .9

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)


    # define the optimizer
    #optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
    #optimizer = tf.train.AdagradOptimizer(lr)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)


def calc_reward(cell_output, br_out, selected_sbs, onehot_labels, raw_labels):
    """
    construct the action network, calculate the reward and finally get the loss
    :param cell_output: cell_output list. length = num_steps, shape = (batch_size, cell_out_size)
    :param br_out: list. length = num_steps, shape = (batch_size, num_sbs)
    :param selected_sbs: list. length = num_steps, shape = (batch_size, num_sbs)
    :param onehot_labels: list. length = num_steps, shape = (batch_size, num_steps, 2)
    :param raw_labels: list. length = num_steps, shape = (batch_size, num_step, 1)
    :return:
    """

    Wb_h_b = tf.get_variable("baselineNet_wts_hiddenState_baseline", (cell_out_size, 1))
    Bb_h_b = tf.get_variable("baselineNet_bias_hiddenState_baseline", (1,1))
    baseline = tf.sigmoid(tf.matmul(cell_output[0], Wb_h_b) + Bb_h_b)
    baselines = tf.tile(baseline, [1, num_sbs])
    no_grad_b = tf.stop_gradient(baselines)
    cell_output = tf.stack(cell_output, 1)
    cell_output = tf.reshape(cell_output, (-1, cell_output.get_shape().as_list()[-1]))

    # get the action(classification)
    p_y = tf.nn.softmax(affine_transform(cell_output, 2, name="soft_max"))
    max_p_y = tf.expand_dims(tf.arg_max(p_y, 1), 1)
    correct_y = tf.cast(tf.reshape(raw_labels, (batch_size*num_steps, 1)), tf.int64)

    # reward for all examples in the batch
    R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
    R = tf.reshape(R, (batch_size, num_steps, 1))
    R = tf.reduce_mean(R, 1)
    reward = tf.reduce_mean(R)
    R = tf.tile(R, [1, num_sbs])  # shape = (batch_size, num_sbs)

    # get the probability from sampled bernoulli process
    p_br = bernoulli_pmf(br_out[0], tf.squeeze(selected_sbs[0])) # calculate the probability of bernoulli process of first num step.

    # define the cost function
    onehot_labels = tf.reshape(onehot_labels, (batch_size*num_steps, 2))

    ce = tf.log(p_y + SMALL_NUM) * tf.cast(onehot_labels, tf.float32)
    ce = tf.reduce_mean(tf.reshape(ce, (batch_size, num_steps, 2)), 1)  # cross entropy loss(ce)
    p_br = tf.stop_gradient(p_br)
    #baselines = tf.stop_gradient(baselines)
    rl = tf.log(p_br + SMALL_NUM) * (R - no_grad_b)  # reinforcement learning loss(rl)
    J = tf.concat(axis=1, values=[ce, rl])  # hybrid loss
    #J = ce
    J = tf.reduce_sum(J, 1)
    J = J - tf.reduce_sum(tf.square(R - baselines), 1)
    J = tf.reduce_mean(J, 0)
    cost = -J

    return cost, reward, tf.squeeze(selected_sbs[0])


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def evaluation(m_valid, valid_data_set, sess, num_batches=100):

    # num_samples = valid_data_set.num_samples
    # num_batches = num_samples / batch_size
    avg_valid_cost = 0.
    avg_valid_reward = 0.
    for i in range(int(num_batches)):

        valid_inputs, valid_labels = valid_data_set.next_batch(batch_size)
        valid_inputs /= fft_size
        # print(train_labels.shape[0])
        valid_onehot_labels = dense_to_one_hot(valid_labels.reshape(-1, 1))
        valid_onehot_labels = valid_onehot_labels.reshape(-1, num_steps, 2)
        feed_dict = {m_valid.inputs: np.expand_dims(valid_inputs, axis=3), m_valid.raw_labels: valid_labels,
                     m_valid.onehot_labels: valid_onehot_labels, m_valid.keep_probability: 1}

        valid_cost, valid_reward = sess.run([m_valid.cost, m_valid.reward], feed_dict=feed_dict)

        avg_valid_cost += valid_cost
        avg_valid_reward += valid_reward

    avg_valid_cost /= (i + 1)
    avg_valid_reward /= (i + 1)

    return avg_valid_cost, avg_valid_reward


class Model(object):

    def __init__(self, is_training=True):

        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.inputs = inputs = tf.placeholder(tf.float32, shape=[batch_size, num_steps, fft_size, channel],
                                              name="inputs")
        self.onehot_labels = onehot_labels = tf.placeholder(tf.int32, shape=[batch_size, num_steps, 2],
                                                            name="onehot_labels")
        self.raw_labels = raw_labels = tf.placeholder(tf.int32, shape=[batch_size, num_steps, 1],
                                                      name="raw_labels")
        inputs = tf.unstack(inputs, axis=1)  # list length = num_steps, shape = (batch_size, fft_size, channel)

        # set inference graph
        cell_output, selected_sbs, br_out = inference(inputs, self.keep_probability, is_training=is_training)
        # set objective function
        self.cost, self.reward, self.selected_sbs = cost, reward, selected_sbs\
            = calc_reward(cell_output, br_out, selected_sbs, onehot_labels, raw_labels)
        # set training strategy
        trainable_var = tf.trainable_variables()
        self.train_op = train(cost, trainable_var)


def main(argv=None):
    #                               Graph Part                               #
    print("Graph initialization...")
    with tf.device(device):
        with tf.variable_scope("model", reuse=None):
            m_train = Model(is_training=True)
        with tf.variable_scope("model", reuse=True):
            m_valid = Model(is_training=False)
    print("Done")
    #                               Summary Part                             #
    with tf.variable_scope("summaries"):
        train_summary_writer = tf.summary.FileWriter(logs_dir + '/train/', max_queue=2)
        train_summary_list = [tf.summary.scalar("cost", m_train.cost), tf.summary.scalar("reward", m_train.reward)]
        train_summary_op = tf.summary.merge(train_summary_list)  # training summary

        avg_valid_cost = tf.placeholder(dtype=tf.float32)
        avg_valid_reward = tf.placeholder(dtype=tf.float32)
        valid_summary_writer = tf.summary.FileWriter(logs_dir + '/valid/', max_queue=2)
        valid_summary_list = [tf.summary.scalar("cost", avg_valid_cost), tf.summary.scalar("reward", avg_valid_reward)]
        valid_summary_op = tf.summary.merge(valid_summary_list)  # validation summary
    #                               Model Save Part                             #
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")
    #                               Session Part                             #
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:  # model restore
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization

    data_set = dr.DataReader(input_dir, output_dir, num_steps=num_steps,
                             name="train")  # training data reader initialization
    # data_set._num_file = 5
    valid_data_set = dr.DataReader(valid_input_dir, valid_output_dir,
                                   num_steps=num_steps, name="valid")  # validation data reader initialization

    for itr in range(max_epoch):
        train_inputs, train_labels = data_set.next_batch(batch_size)
        train_inputs /= fft_size
        train_onehot_labels = dense_to_one_hot(train_labels.reshape(-1, 1))
        train_onehot_labels = train_onehot_labels.reshape(-1, num_steps, 2)
        feed_dict = {m_train.inputs: np.expand_dims(train_inputs, axis=3), m_train.raw_labels: train_labels,
                     m_train.onehot_labels: train_onehot_labels, m_train.keep_probability: dropout_rate}

        sess.run(m_train.train_op, feed_dict=feed_dict)

        if itr % 20 == 0:
            train_cost, train_reward, train_summary_str = sess.run([m_train.cost, m_train.reward, train_summary_op],
                                                                   feed_dict=feed_dict)
            print("Step: %d, train_cost: %.5f, train_reward: %.5f" % (itr, train_cost, train_reward))

            train_summary_writer.add_summary(train_summary_str, itr)  # write the train phase summary to event files

        if itr % 500 == 0:
            saver.save(sess, logs_dir + "/model.ckpt", itr)  # model save

            valid_cost, valid_reward = evaluation(m_valid, valid_data_set, sess, num_valid_batches)

            print("valid_cost: %.5f, valid_reward: %.5f" % (valid_cost, valid_reward))

            valid_summary_str = sess.run(valid_summary_op, feed_dict={avg_valid_cost: valid_cost,
                                                                      avg_valid_reward: valid_reward})
            valid_summary_writer.add_summary(valid_summary_str, itr)
if __name__ == "__main__":
    tf.app.run()


