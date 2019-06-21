# -*- coding: utf-8 -*-
#

import sys
import argparse

import chainer
import chainer.optimizer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np

import model
import utils
import updater as my_updater
import denoise

PAD = 0
GO = 1
EOS = 2
UNK = 3

class MapDict:
    pass

opt = MapDict()
opt.permutation = 0
opt.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

def convert(batch, device):
    if device < 0:
        xp = np
    else:
        xp = cuda.cupy
    x = [x_ for x_ in batch]
    x = denoise.add_noise(x, opt)
    x_org = [x_ for x_ in batch]
    x = utils.prepare_data_for_cnn(x, opt.maxlen, opt.filter_shape)
    x_org = utils.prepare_data_for_rnn(x_org, opt.maxlen, opt.sent_len, opt.n_words, is_add_GO=True)
    x = xp.array(x, dtype=np.int32)
    x_org = xp.array(x_org, dtype=np.int32)
    return {'x': x, 'x_org': x_org}

def load_vocaburaly(path):
    with open(path) as f:
        word_ids = {line.strip(): i + 4 for i, line in enumerate(f)}
        word_ids['<PAD>'] = 0
        word_ids['<GO>'] = 1
        word_ids['<EOS>'] = 2
        word_ids['<UNK>'] = 3
        return word_ids

def load_data(vocabulary, file, add_EOF=False):
    data = []
    print('loading...: %s' % file)
    with open(file) as f:
        for line in f:
            words = line.strip().split()
            if add_EOF:
                words.append('<EOS>')
            array = np.array([vocabulary.get(w, UNK) for w in words], dtype=np.int32)
            data.append(array)
    return data

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('vocab')
    p.add_argument('--test', type=str, default="")
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--batchsize', '-b', type=int, default=64)
    p.add_argument('--epoch', '-e', type=int, default=100)
    p.add_argument('--out', '-o', default='result')
    p.add_argument('--log-interval', type=int, default=100)
    p.add_argument('--valid-interval', type=int, default=200)
    p.add_argument('--resume', '-r', default=None)
    p.add_argument('--snap-interval', type=int, default=1000,
                        help='number of iteration to take snapshot')
    args = p.parse_args()
    return args

def train(args):
    word_ids = load_vocaburaly(args.vocab)
    source = load_data(word_ids, args.source)
    if args.test == "":
        test = load_data(word_ids, args.source)
    else:
        test = load_data(word_ids, args.test)
    opt.filter_shape = 5
    opt.maxlen = 51
    opt.sent_len = opt.maxlen + 2*(opt.filter_shape-1)
    opt.n_words = len(word_ids)
    train_data = source
    test_data = test
    source_words = {i: w for w, i in word_ids.items()}

    report_keys = ['epoch', 'iteration', "loss_dis", "loss_gen", 'elapsed_time']

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=True)

    models = []
    opts = {}
    updater_args = {
        "iterator": {'main': train_iter},
        "device": args.gpu,
        "opt": opt
    }
