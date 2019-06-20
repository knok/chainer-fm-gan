# -*- coding: utf-8 -*-
#

import sys
import argparse

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np

sys.path.append('./chainer')
import model
import utils
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
        #word_ids['<GO>'] = len(word_ids) + 1
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

    mdl = model.auto_encoder(opt.n_words, opt.maxlen)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        mdl.to_gpu(args.gpu)
    
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(mdl)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=True)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport( \
        ['epoch', 'iteration', 'main/loss', 'main/kl_loss', 'elapsed_time']), \
            trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.PrintReport( \
        ['epoch', 'iteration', 'main/loss', 'main/kl_loss', 'elapsed_time']), \
            trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.snapshot(), 
        trigger=(args.snap_interval, 'iteration'))
    # validation/evaluation
    @chainer.training.make_extension()
    def translate(trainer):
        test = convert(test_iter.next(), args.gpu)
        _, sent1, sent2 = mdl.forward(test['x'], test['x_org'])
        result = sent2[0]
        source_sentence = ' '.join([source_words[int(i)] for i in test['x_org'][0] if i != PAD])
        sent1_results = ' '.join([source_words.get(int(i), '*NOKEY') for i in sent1[0].data if i != PAD])
        result_sentence = ' '.join([source_words.get(int(i), '*NOKEY') for i in result.data if i != PAD])
        print('# source(p) : ' + source_sentence)
        print('# sent1     : ' + sent1_results)
        print('# sent2     : ' + result_sentence)
    trainer.extend(
        translate, trigger=(args.valid_interval, 'iteration'))

    print('start training')

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    
    trainer.run()

    chainer.serializers.save_npz(args.out + '/finalmodel.npz', mdl)

def main():
    args = get_args()
    train(args)

if __name__ == '__main__':
    main()