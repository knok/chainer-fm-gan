# -*- coding: utf-8 -*-

import numpy as np

PAD = 0
GO = 1
EOS = 2
UNK = 3

def prepare_data_for_cnn(seqs_x, maxlen, filter_shape):
    lengths_x = [len(s) for s in seqs_x]
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
    pad = filter_shape - 1
    x = []
    for rev in seqs_x:
        xx = []
        for i in range(pad):
            xx.append(0) # PAD
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x, dtype=np.int32)
    return x

def prepare_data_for_rnn(seqs_x, maxlen, sent_len, n_words, is_add_GO=True):
    lengths_x = [len(s) for s in seqs_x]
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros((n_samples, sent_len), dtype=np.int32)
    for idx, s_x in enumerate(seqs_x):
        if is_add_GO:
            x[idx, 0] = GO
            x[idx, 1:lengths_x[idx]+1] = s_x
        else:
            x[idx, :lengths_x[idx]] = s_x
    return x