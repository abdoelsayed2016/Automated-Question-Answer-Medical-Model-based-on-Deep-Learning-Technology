import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import argparse
import time,math,random
from sklearn.model_seection import train_test_split
import torch
import torch.nn as nn

from torch import optim

import data as dataloader

import seq2seq


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def argparser():
    p=argparse.ArgumentParser()

    p.add_argument('--num_iters',required=True,type=int)
    p.add_argument('--embedding_size',type=int,default=300)
    p.add_argument('--hidden_size',type=int,default=300)

    configuration=p.parse_args()
    return  configuration


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

def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

