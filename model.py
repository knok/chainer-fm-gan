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
        self.lstm.reset_state()
        self.lstm.h = initial_state[0]
        self.lstm.c = initial_state[1]
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop and prev is not None:
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

class auto_encoder(chainer.Chain):
    def __init__(self, n_words, maxlen=51, embed_size=300, n_gan=128, ac_func=F.tanh, stride=2, ef_dim=128, n_hid=100,
            trainable=True, relu=False):
        super().__init__()
        self.ef_dim = ef_dim
        with self.init_scope():
            self.embedding = embedding(n_words, embed_size, trainable, relu)
            self.conv_encoder = conv_encoder(n_gan, ac_func, stride, maxlen=maxlen)
            self.vae_classifier = vae_classifier_2layer(ef_dim)
            self.lstm_decoder = lstm_decoder_embedding(self.embedding, n_hid, n_words, embed_size)

    def forward(self, x, x_org, feed_previous=False):
        bsize = len(x)
        x_emb = self.embedding(x)
        x_emb = normalizing(x_emb, 1)
        x_emb = F.expand_dims(x_emb, 1)
        H = self.conv_encoder(x_emb)
        H_mean, H_log_sigma_sq = self.vae_classifier(H)
        mu = self.xp.zeros((bsize, self.ef_dim), np.float32)
        ln_sigma = self.xp.ones((bsize, self.ef_dim), np.float32)
        eps = F.gaussian(mu, ln_sigma) # N(0, 1)
        H_dec = H_mean + eps * F.sqrt(F.exp(H_log_sigma_sq))
        H_dec2 = F.identity(H_dec)
        # moddel: cnn_rnn
        loss, rec_sent_1, _ = self.lstm_decoder(H_dec2, x_org, feed_previous=feed_previous)
        _, rec_sent_2, _ = self.lstm_decoder(H_dec2, x_org, feed_previous=True)
        # KL loss
        kl_loss = F.mean(-0.5 * F.mean(1 + H_log_sigma_sq \
                                    - F.square(H_mean) \
                                    - F.exp(H_log_sigma_sq), axis=1))
        loss += kl_loss
        chainer.report({'loss': loss.data, 'kl_loss': kl_loss.data}, self)
        return loss, rec_sent_1, rec_sent_2

    def __call__(self, x, x_org):
        loss, _, _ = self.forward(x, x_org)
        return loss

class textGan_generator(chainer.Chain):
    def __init__(self, n_words, maxlen=51, embed_size=300, n_gan=128, ac_func=F.tanh, stride=2, \
            ef_dim=128, n_hid=100, trainable=True, relu=False, fsize=300, filter_shape=5):
        super().__init__()
        self.L = n_hid
        self.ef_dim = n_gan
        with self.init_scope():
            self.embedding = embedding(n_words, embed_size, trainable, relu)
            self.lstm_decoder = lstm_decoder_embedding(self.embedding, n_hid, n_words, embed_size)

    def make_hidden(self, bsize):
        mu = self.xp.zeros((bsize, self.ef_dim), np.float32)
        ln_sigma = self.xp.ones((bsize, self.ef_dim), np.float32)
        z = F.gaussian(mu, ln_sigma)
        return z

    def __call__(self, x, x_org):
        bsize = len(x)
        z = self.make_hidden(bsize)
        # lstm
        x_emb, _ = self.embedding(x)
        x_emb = F.expand_dims(x_emb, 1)
        x_emb = normalizing(x_emb, 1)
        _, syn_sents, logits = self.lstm_decoder(z, x_org, feed_previous=True)
        prob = [F.softmax(l * self.L) for l in logits]
        prob = F.stack(prob, 1)

        return syn_sents, prob

class discriminator_2layer(chainer.Chain):
    def __init__(self):
        super().__init__()
        opt_H_dis = 300
        biasInit = chainer.initializers.Constant(0.001, dtype=np.float32)
        # opt.tanh = None, opt.batch_nrom = False
        with self.init_scope():
            self.fc1 = L.Linear(None, opt_H_dis, initial_bias=biasInit)
            self.fc2 = L.Linear(None, 2, initial_bias=biasInit)
    def __call__(self, H):
        H = F.squeeze(H)
        # regularization - dropout only
        H_reg = F.dropout(H)
        H_dis = F.relu(self.fc1(H_reg))
        H_dis = F.dropout(H_dis)
        logits = self.fc2(H_dis)
        return logits

class discriminator(chainer.Chain):
    def __init__(self, embedding, n_gan, ac_func, stride, maxlen, fsize, filter_shape, embed_size):
        super().__init__()
        with self.init_scope():
            self.embedding = embedding
            self.encoder = conv_encoder(n_gan, ac_func, stride, maxlen=maxlen)
            self.disc = discriminator_2layer()
    def __call__(self, x, is_prob=False):
        if is_prob:
            x_emb = F.tensordot(x, self.embedding.embed.W, [[2],[0]])
        else:
            x_emb, _ = self.embedding(x)
        x_emb = F.expand_dims(x_emb, 1)
        H = self.encoder(x_emb)
        logits = self.disc(H)
        return logits, H

class textGan_discriminator(chainer.Chain):
    def __init__(self, embedding, n_words, maxlen=51, embed_size=300, n_gan=128, \
            ac_func=F.tanh, stride=2, ef_dim=128, n_hid=100, \
            trainable=True, relu=False, fsize=300, filter_shape=5):
        super().__init__()
        with self.init_scope():
                self.embedding = embedding
                self.disc = discriminator(self.embedding, n_gan, ac_func, stride, maxlen, fsize, filter_shape, embed_size)

    def __call__(self, inp, is_prob=False):
        logits, H = self.disc(inp, is_prob=is_prob)
        return logits, H

def main():
    import utils
    maxlen = 51
    filter_shape = 5
    sent_len = maxlen + 2*(filter_shape-1)
    n_words = 5728
    #m = auto_encoder(n_words, maxlen=maxlen)
    m = textGan_generator(n_words, maxlen)
    d = textGan_discriminator(m.embedding, n_words, maxlen=maxlen)
    # m = textGan(n_words, maxlen=maxlen)
    data = np.arange(20*2, dtype=np.int32).reshape(2, 20)
    x = utils.prepare_data_for_cnn(data, maxlen, filter_shape)
    x_orig = utils.prepare_data_for_rnn(data, maxlen, sent_len, n_words)
    syn_sents, prob = m(x, x_orig)
    # l, h = d(x)
    # l, h = d(prob, is_prob=True)

if __name__== '__main__':
    main()