# -*- coding: utf-8 -*-
"""FM-GAN model
"""

import math
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

def normalizing(x, axis):
    square = x * x
    sq_sum = F.sum(square, axis=axis, keepdims=True)
    norm = F.sqrt(sq_sum)
    normalized = x / norm
    return normalized

def compute_MMD_loss(H_fake, H_real):
    sigma_range = [2]
    dividend = 1
    bsize, dim = H_fake.shape
    dist_x, dist_y = H_fake/dividend, H_real/dividend
    x_sq = F.expand_dims(F.sum(dist_x ** 2, axis=1), 1)
    y_sq = F.expand_dims(F.sum(dist_y ** 2, axis=1), 1)
    dist_x_T = F.transpose(dist_x)
    dist_y_T = F.transpose(dist_y)
    x_sq_T = F.transpose(x_sq)
    y_sq_T = F.transpose(y_sq)

    tempxx = -2 * F.matmul(dist_x, dist_x_T) + x_sq + x_sq_T
    tempxy = -2 * F.matmul(dist_x, dist_y_T) + x_sq + y_sq_T
    tempyy = -2 * F.matmul(dist_y, dist_y_T) + y_sq + y_sq_T

    for sigma in sigma_range:
        kxx, kxy, kyy = 0, 0, 0
        kxx += F.mean(F.exp(-tempxx / 2 / (sigma ** 2)))
        kxy += F.mean(F.exp(-tempxy / 2 / (sigma ** 2)))
        kyy += F.mean(F.exp(-tempyy / 2 / (sigma ** 2)))

    gan_cost_g = F.sqrt(kxx + kyy - 2*kxy)
    return gan_cost_g

class embedding(chainer.Chain):
    """embedding with normalization"""
    def __init__(self, n_words, embed_size, trainable, relu=False):
        super().__init__()
        emb_init = chainer.initializers.Uniform(scale=0.001)
        self.relu = relu
        with self.init_scope():
            self.embed = L.EmbedID(n_words, embed_size, initialW=emb_init)
        if not trainable:
            self.embed.disable_update()
    
    def __call__(self, features):
        W = self.embed(features)
        if self.relu:
            W = F.relu(W)
        return W

class conv_encoder(chainer.Chain):
    def __init__(self, n_gan, act_func, stride=2, fsize=300, filter_shape=5,
     embed_size=300, maxlen=50):
        super().__init__()
        self.act_func = act_func
        self.sent_len = maxlen + 2*(filter_shape-1)
        self.sent_len2 = int(math.floor((self.sent_len - filter_shape) / stride) + 1)
        self.sent_len3 = int(math.floor((self.sent_len2 - filter_shape)/stride) + 1)
        initw = chainer.initializers.Constant(0.001, dtype=np.float32)
        with self.init_scope():
            self.bnorm1 = L.BatchNormalization(1, decay=0.9)
            self.conv1 = L.Convolution2D(None, fsize, ksize=(filter_shape, embed_size),
            stride=(stride, 1), initialW=initw, pad=0)
            self.bnorm2 = L.BatchNormalization(fsize, decay=0.9)
            self.conv2 = L.Convolution2D(None, fsize*2, ksize=[filter_shape, 1], 
            stride=(stride, 1), pad=0) # VALID
            self.bnorm3 = L.BatchNormalization(fsize*2, decay=0.9)
            self.conv3 = L.Convolution2D(None, n_gan, ksize=[self.sent_len3, 1], pad=0)
    
    def __call__(self, x):
        X = self.bnorm1(x)
        h1 = self.conv1(X)
        h1 = self.bnorm2(h1)
        h2 = self.conv2(h1)
        h2 = self.bnorm3(h2)
        h3 = self.conv3(h2)
        if self.act_func:
            h3 = self.act_func(h3)
        return h3

class vae_classifier_2layer(chainer.Chain):
    def __init__(self, ef_dim=128):
        super().__init__()
        self.dropout_ratio = 0
        self.ef_dim = ef_dim
        initbias = chainer.initializers.Constant(0.001, dtype=np.float32)
        with self.init_scope():
            self.fc1 = L.Linear(None, ef_dim, initial_bias=initbias)
            self.fc2 = L.Linear(None, ef_dim, initial_bias=initbias)
            self.fc3 = L.Linear(None, ef_dim, initial_bias=initbias)
    
    def __call__(self, h):
        #h = F.squeeze(h)
        bsize, wd, ht, ch = h.shape
        h = h.reshape(bsize, wd, ht)
        H_dis = F.relu(self.fc1(F.dropout(h, self.dropout_ratio)))
        mean = self.fc2(F.dropout(H_dis, self.dropout_ratio))
        log_sigma_sq = self.fc3(F.dropout(H_dis, self.dropout_ratio))
        return mean, log_sigma_sq

class lstm_decoder_embedding(chainer.Chain):
    def __init__(self, shared_emb, n_hid, n_words, embed_size=300):
        super().__init__()
        self.dropout_ratio = 0
        self.n_words = n_words
        initbias = chainer.initializers.Constant(0.001, dtype=np.float32)
        initw = chainer.initializers.Uniform(scale=0.001, dtype=np.float32)
        with self.init_scope():
            self.embed = shared_emb
            self.fc1 = L.Linear(None, n_hid, initial_bias = initbias)
            self.fc11 = L.Linear(None, embed_size, initial_bias = initw, initialW=initw)
            self.lstm = L.LSTM(embed_size + n_hid, n_hid)
            self.W = chainer.Parameter(initializer=initw, shape=[n_hid, embed_size])
            self.b = chainer.Parameter(initializer=initw, shape=[n_words])
            self.fc3 = L.Linear(n_hid, n_words, initialW=initw)

    def loop_function(self, prev, h, output_ptojection=False):
        if output_ptojection:
            prev = prev * self.W + self.b
        prev_symbol = F.argmax(prev, 1)
        emb_prev = F.embed_id(prev_symbol, normalizing(self.embed.W, 1))
        emb_prev = F.concat([emb_prev, h], 1)
        return emb_prev

    def rnn_decoder_truncated(self, decoder_inputs, initial_state, feed_previous, \
        loop=False):
        self.lstm.h = initial_state[0]
        self.lstm.c = initial_state[1]
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop:
                inp = self.loop_function(prev, initial_state[0], loop)
            ys = self.lstm(inp)
            state = (self.lstm.h, self.lstm.c)
            output = F.vstack(ys)
            outputs.append(output)
        return outputs, state

    def __call__(self, H, y, feed_previous=False):
        bsize = H.shape[0]
        y = F.stack(y, axis=1)
        H0 = self.fc1(H)
        c = self.xp.zeros_like(H0, dtype=np.float32)
        H1 = (H0, c)
        y_input = []
        for features in y:
            x_emb = self.embed(features)
            x_emb = normalizing(x_emb, 1)
            y_input.append(F.concat([x_emb, H0], 1))
        out_proj = True if feed_previous else False
        outputs, state = self.rnn_decoder_truncated(y_input, H1, feed_previous, \
            out_proj)
        # logits
        logits = [self.fc3(out) for out in outputs]
        syn_sents = [F.argmax(l, 1) for l in logits]
        syn_sents = F.stack(syn_sents, 1)

        # loss
        seq_logits = F.concat(F.stack(logits[:-1], 1), 0)
        concat_ys_out = F.concat(F.stack(y[1:], 1), 0)
        loss = F.sum(F.softmax_cross_entropy(seq_logits, concat_ys_out, reduce='no')) \
            / bsize
        return loss, syn_sents, logits
