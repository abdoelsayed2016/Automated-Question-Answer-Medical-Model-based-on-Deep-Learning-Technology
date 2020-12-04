import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import argparse
import time, math, random
from sklearn.model_seection import train_test_split
import torch
import torch.nn as nn

from torch import optim

import data as dataloader

import seq2seq

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--num_iters', required=True, type=int)
    p.add_argument('--embedding_size', type=int, default=300)
    p.add_argument('--hidden_size', type=int, default=300)

    configuration = p.parse_args()
    return configuration


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


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentences(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(dataloader.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorFromPair(pair):
    input_tensor = tensorFromSentences(input_lang, pair[0])
    target_tensor = tensorFromSentences(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=dataloader.MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length, target_length = input_tensor.size(0), target_tensor.size(0)

    encoder_hidden = encoder.initHidden().to(device)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]


if __name__ == "__main__":
    config = argparser()
    teacher_focing_ratio = 0.5

    input_lang, output_lang, pairs = dataloader.perpare_dataset('q', 'a')

    encoder = seq2seq.GRUEncodeer(input_size=input_lang.num_words, embedding_size=config.embedding_size,
                                  hidden_size=config.hidden_size).to(device)

    decoder = seq2seq.GRUDecoder(output_size=output_lang.num_words,
                                 embedding_size=config.embedding_size,
                                 hidden_size=config.hidden_size).to(device)

    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))
