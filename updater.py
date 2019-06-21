# -*- coding: utf-8 -*-

import sys

import numpy as np
import chainer
import chainer.functions as F

from model import compute_MMD_loss
import denoise
import utils

class FmGanUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.opt = kwargs.pop('opt')
        super(FmGanUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        
        xp = self.gen.xp
        opt = self.opt

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = denoise.add_noise(batch, self.opt)
        x = utils.prepare_data_for_cnn(x, opt.maxlen, opt.filter_shape)
        x_org = utils.prepare_data_for_rnn(batch, opt.maxlen, opt.sent_len, opt.n_words, is_add_GO=True)
        x = xp.array(x, dtype=np.int32)
        x_org = xp.array(x_org, dtype=np.int32)
        # generator
        syn_sents, prob = self.gen(x, x_org) # prob: fake data

        # discriminator
        logits_real, H_real = self.dis(x)
        logits_fake, H_fake = self.dis(prob, is_prob=True)

        # one hot vector
        labels_one = xp.ones((batchsize), dtype=xp.int32) # 1-dim array
        labels_zero = xp.zeros((batchsize), dtype=xp.int32)
        labels_fake = labels_zero #F.concat([labels_one, labels_zero], axis=1)
        labels_real = labels_one #F.concat([labels_zero, labels_one], axis=1)
        D_loss = F.softmax_cross_entropy(logits_real, labels_real) + \
            F.softmax_cross_entropy(logits_fake, labels_fake)

        G_loss = compute_MMD_loss(F.squeeze(H_fake), F.squeeze(H_real))

        self.gen.cleargrads()
        G_loss.backward()
        gen_optimizer.update()

        self.dis.cleargrads()
        D_loss.backward()
        dis_optimizer.update()

        H_fake.unchain_backward()
        H_real.unchain_backward()
        prob.unchain_backward()

        chainer.reporter.report({'loss_gen': G_loss})
        chainer.reporter.report({'loss_dis': D_loss})