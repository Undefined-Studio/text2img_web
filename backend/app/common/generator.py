import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import os
import re
import pickle
import string
import nltk

# Generator相关参数Map
Generator_Param_Map = {
    'checkpoint_dir': 'checkpoint',
    'vocab_dir': '_vocab.pickle',
    'batch_size': 64,
    'image_size': 64,
    'z_dim': 512
}


class Generator:
    def __init__(self):
        batch_size = Generator_Param_Map['batch_size']
        image_size = Generator_Param_Map['image_size']
        checkpoint_dir = Generator_Param_Map['checkpoint_dir']
        vocab_dir = Generator_Param_Map['vocab_dir']
        z_dim = Generator_Param_Map['z_dim']

        self.sample_size = batch_size

        # 边长
        self.ni = int(np.ceil(np.sqrt(batch_size)))

        with open(vocab_dir, 'rb') as f:
            self.vocab = pickle.load(f)

        self.t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='real_image')

        self.t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')

        self.t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

        net_rnn = rnn_embed(self.t_real_caption, is_train=False, reuse=False)
        net_g, _ = generator_txt2img_resnet(self.t_z,
                                            net_rnn.outputs,
                                            is_train=False, reuse=False, batch_size=batch_size)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tl.layers.initialize_global_variables(self.sess)

        net_rnn_name = os.path.join(checkpoint_dir, 'net_rnn.npz100.npz')
        net_cnn_name = os.path.join(checkpoint_dir, 'net_cnn.npz100.npz')
        net_g_name = os.path.join(checkpoint_dir, 'net_g.npz100.npz')
        net_d_name = os.path.join(checkpoint_dir, 'net_d.npz100.npz')

        self.net_rnn_res = tl.files.load_and_assign_npz(sess=self.sess, name=net_rnn_name, network=net_rnn)

        self.net_g_res = tl.files.load_and_assign_npz(sess=self.sess, name=net_g_name, network=net_g)

        sample_size = batch_size
        self.sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

    def run(self, sentences):
        sample = [sentences[0]] * int(self.sample_size/self.ni)
        for sentence in sentences[1:]:
            sample += ([sentence] * int(self.sample_size/self.ni))

        for i, sentence in enumerate(sample):
            sentence = preprocess_caption(sentence)
            sample[i] = [self.vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [
                self.vocab.end_id]

        sample = tl.prepro.pad_sequences(sample, padding='post')
        img_gen, rnn_out = self.sess.run([self.net_g_res.outputs, self.net_rnn_res.outputs], feed_dict={
            self.t_real_caption: sample,
            self.t_z: self.sample_seed})

        save_images(img_gen, [self.ni, self.ni], 'app/static/gen.png')


def rnn_embed(input_seqs, is_train=True, reuse=False, return_embed=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
            inputs=input_seqs,
            # vocab_size = 8000
            vocabulary_size=8000,
            # word_embedding_size = 256
            embedding_size=256,
            E_init=w_init,
            name='rnn/wordembed')
        network = DynamicRNNLayer(network,
                                  cell_fn=LSTMCell,
                                  cell_init_args={'state_is_tuple': True, 'reuse': reuse},
                                  # for TF1.1, TF1.2 dont need to set reuse
                                  # rnn_hidden_size = 128
                                  n_hidden=128,
                                  # keep_prob = 1.0
                                  dropout=(1 if is_train else None),
                                  initializer=w_init,
                                  sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
                                  return_last=True,
                                  name='rnn/dynamic')
        return network


def generator_txt2img_resnet(input_z, t_txt=None, is_train=True, reuse=False, batch_size=64):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # image_size = 64
    s = 64
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='g_input_txt')
            # t_dim = 128
            net_txt = DenseLayer(net_txt, n_units=128,
                                 act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

        net_h0 = DenseLayer(net_in, gf_dim * 8 * s16 * s16, act=tf.identity,
                            W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0,  # act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim * 8], name='g_h0/reshape')

        net = Conv2d(net_h0, gf_dim * 2, (1, 1), (1, 1),
                     padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h1_res/batch_norm')
        net = Conv2d(net, gf_dim * 2, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        net = Conv2d(net, gf_dim * 8, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        net = BatchNormLayer(net,  # act=tf.nn.relu,
                             is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        net_h1 = ElementwiseLayer(layer=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
        net_h1.outputs = tf.nn.relu(net_h1.outputs)

        # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
        # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
                                   align_corners=False, name='g_h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim * 4, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,  # act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                     padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h3_res/batch_norm')
        net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        net = Conv2d(net, gf_dim * 4, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        net = BatchNormLayer(net,  # act=tf.nn.relu,
                             is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_h3/add')
        net_h3.outputs = tf.nn.relu(net_h3.outputs)

        # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d'),
        net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
                                   align_corners=False, name='g_h4/upsample2d')
        net_h4 = Conv2d(net_h4, gf_dim * 2, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h5/decon2d')
        net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
                                   align_corners=False, name='g_h5/upsample2d')
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        # net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        net_ho = UpSampling2dLayer(net_h5, size=[s, s], is_scale=False, method=1,
                                   align_corners=False, name='g_ho/upsample2d')
        # c_dim = 3
        net_ho = Conv2d(net_ho, 3, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho, logits


def preprocess_caption(line):
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    return prep_line


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def save_images(images, size, image_path):
    return imsave(images, size, image_path)
