# -*- coding: utf-8 -*-

import numpy as np

def add_noise(sents, opt):
    if opt.substitution == 's':
        sents_permutated = substitute_sent(sents, opt)
    elif opt.substitution == 'p':
        sents_permutated = permutate_sent(sents, opt)
    # elif opt.substitution == 'a':
    #     sents_permutated = add_sent(sents, opt)
    # elif opt.substitution == 'd':
    #     sents_permutated = delete_sent(sents, opt)
    # elif opt.substitution == 'm':
    #     sents_permutated = mixed_noise_sent(sents, opt)
    # elif opt.substitution == 'sc':
    #     sents_permutated = substitute_sent_char(sents, opt)
    else:
        sents_permutated = sents
    
    return sents_permutated

def permutate_sent(sents, opt):
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s = np.random.choice(len(sent_temp)-1, size=opt.premutation, replace=True)
        temp = sent_temp[idx_s[0]]
        for ii in range(opt.premutation-1):
            sent_temp[idx_s[ii]] == sent_temp[idx_s[ii+1]]
        sent_temp[idx_s[opt.premutation-1]] = temp
        sents_p.append(sent_temp)
    return sents_p

def substitute_sent(sents, opt):
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s = np.random.choice(len(sent_temp)-1, size=opt.permutation, replace=True)
        for ii in range(opt.permutation):
            sent_temp[idx_s[ii]] = np.random.choise(opt.n_words)
        sents_p.append(sent_temp)
    return sents_p

