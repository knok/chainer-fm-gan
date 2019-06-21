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
    p.add_argument('--out', '-o', default='result-gan')
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

    generator = model.textGan_generator(opt.n_words, maxlen=opt.maxlen)
    discriminator = model.textGan_discriminator( \
        generator.embedding, opt.n_words)
    models = [generator, discriminator]
    updater_args["models"] = models

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        for mdl in models:
            mdl.to_gpu(args.gpu)
    
    opts["opt_gen"] = chainer.optimizers.Adam()
    opts["opt_gen"].setup(generator)
    opts["opt_gen"].add_hook(chainer.optimizer.GradientClipping(5.0))
    opts["opt_dis"] = chainer.optimizers.Adam()
    opts["opt_dis"].setup(discriminator)
    opts["opt_dis"].add_hook(chainer.optimizer.GradientClipping(5.0))
    updater_args["optimizer"] = opts

    updater = my_updater.FmGanUpdater(**updater_args)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    for m in models:
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snap_interval, 'iteration'))
    trainer.extend(extensions.LogReport( \
        report_keys), trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.PrintReport( \
        report_keys), trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.snapshot(), 
        trigger=(args.snap_interval, 'iteration'))
    # validation/evaluation
    @chainer.training.make_extension()
    def translate(trainer):
        vaild_index = np.random.choice(len(test_data), args.batchsize)
        val_sents = [test[t] for t in vaild_index]
        val_sents_permutated = denoise.add_noise(val_sents, opt)
        x_val_batch = utils.prepare_data_for_cnn(val_sents_permutated, opt.maxlen, opt.filter_shape)
        x_val_batch_org = utils.prepare_data_for_rnn(val_sents, opt.maxlen, opt.sent_len, opt.n_words, is_add_GO=True)
        mdl = generator
        xp = mdl.xp
        x_val_batch = xp.array(x_val_batch)
        x_val_batch_org = xp.array(x_val_batch_org)
        syn_sents, logits = mdl(x_val_batch, x_val_batch_org)
        prob = [F.softmax(l * opt.L) for l in logits]
        prob = F.stack(prob, 1)
        source_sentence = ' '.join([source_words[int(i)] for i in x_val_batch_org[0] if i != PAD])
        result_sentence = ' '.join([source_words.get(int(i), '*NOKEY') for i in syn_sents.data[0] if i != PAD])
        prob_sentence = ' '.join([source_words[xp.argmax(p)] for p in prob.data[:, 0]])
        print('# source : ' + source_sentence)
        print('# sent2  : ' + result_sentence)
        print('# prob   : ' + prob_sentence)
    trainer.extend(
        translate, trigger=(args.valid_interval, 'iteration'))

    #
    print('load pretraining')
    chainer.serializers.load_npz(args.generator_pretrain, generator)

    print('start training')

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    
    trainer.run()

    chainer.serializers.save_npz(args.out + '/final-gen.npz', generator)
    chainer.serializers.save_npz(args.out + '/final-dis.npz', discriminator)

def main():
    args = get_args()
    train(args)

if __name__ == '__main__':
    main()