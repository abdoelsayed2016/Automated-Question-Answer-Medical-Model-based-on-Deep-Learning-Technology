import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import argparse
import time
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim

import data as data

import seq2seq


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def argparser():
    p=argparse.ArgumentParser()
    p.add_argument('--num_iters',required=True,type=int)
    p.add_argument('--embedding_size',type=int,default=300)
    p.add_argument('--hidden_size',type=int,default=300)
    config=p.parse_args()
    return config

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__=='__main__':
    config=argparser()

    teacher_forcing_ratio=0.5

    input_lang,output_lang,pairs=data.perpare_dataset('q','a')

    encoder=seq2seq.BiLSTMEncoder(input_size = input_lang.n_words,
                                 embedding_size = config.embedding_size,
                                 hidden_size = config.hidden_size
                                 ).to(device)

    decoder = seq2seq.BiLSTMDecoder(output_size = output_lang.n_words,
                                 embedding_size = config.embedding_size,
                                 hidden_size = config.hidden_size
                                 ).to(device)

    trainiters(pairs, encoder, decoder, config.n_iters)
