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

def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang,sentence):
    indexes=indexesFromSentence(lang,sentence)
    indexes.append(data.EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorsFromPair(pair):
    input_tensor=tensorFromSentence(input_lang,pair[0])
    target_tensor=tensorFromSentence(output_lang,pair[1])
    return (input_tensor,target_tensor)

def merge_encoder_hiddens(encoder_hiddens):
    new_hiddens,new_cells=[],[]
    hiddens,cells=encoder_hiddens

    for i in range(0,hiddens.size(0),2):
        new_hiddens+=[torch.cat([hiddens[i],hiddens[i+1]],dim=-1)]
        new_cells+=[torch.cat([cells[i],cells[i+1]],dim=-1)]

    new_hiddens,new_cells=torch.stack(new_hiddens),torch.stack(new_cells)

    return new_hiddens,new_cells

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=data.MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length,target_length=input_tensor.size(0),target_tensor.size(0)
    encoder_hidden =(encoder.init_hidden().to(device),encoder.init_hidden().to(device))
    encoder_outputs=torch.zeros(max_length,encoder.hidden_size).to(device)

    loss=0
    for ei in range(input_length):
        encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
        encoder_outputs[ei]=encoder_output[0,0]

    decoder_input=torch.tensor([[data.SOS_token]]).to(device)
    decoder_hidden=merge_encoder_hiddens(encoder_hidden)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input.
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size)
            # 2: respectively, hidden state and cell state.
            # Here, the lstm layer in decoder is uni-directional.

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # teacher forcing
            # |decoder_input|, |target_tensor[di]] = (1)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size)
            # 2: respectively, hidden state and cell state.
            # Here, the lstm layer in decoder is uni-directional.

            topv, topi = decoder_output.topk(1)  # top-1 value, index
            # |topv|, |topi| = (1, 1)

            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            # |target_tensor[di]| = (1)
            if decoder_input.item() == data.EOS_token:
                # |decoder_input| = (1)
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()/target_length


def trainiters(pairs, encoder, decoder, n_iters,
               train_pairs_seed=0, print_every=1000, plot_every=1000, learning_rate=.01):
    start = time.time()
    plot_losses = []
    print_loss_total, plot_loss_total = 0, 0

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    """train_pairs *= n_iters//len(train_pairs)
    train_pairs += [random.choice(train_pairs) for i in range(n_iters%len(train_pairs))]
    train_pairs = [tensorsFromPair(pair) for pair in train_pairs]"""
    train_pairs = [tensorsFromPair(random.choice(pairs))
                   for i in range(n_iters)]
    # |train_pairs| = (n_iters, 2, sentence_length, 1) # eng, fra

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        pair = train_pairs[iter - 1]
        # |pair| = (2) # eng, fra
        input_tensor, target_tensor = pair[0], pair[1]
        # |input_tensor|, |target_tensor| = (sentence_length, 1)

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        # print loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # plot loss
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / print_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

    plt.savefig('baseline-Reverse-loss')
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')

if __name__=='__main__':
    config=argparser()

    teacher_forcing_ratio=0.5

    input_lang,output_lang,pairs=data.perpare_dataset('q','a')

    encoder=seq2seq.BiLSTMEncoder(input_size = input_lang.num_words,
                                 embedding_size = config.embedding_size,
                                 hidden_size = config.hidden_size
                                 ).to(device)

    decoder = seq2seq.BiLSTMDecoder(output_size = output_lang.num_words,
                                 embedding_size = config.embedding_size,
                                 hidden_size = config.hidden_size
                                 ).to(device)

    trainiters(pairs, encoder, decoder, config.n_iters)
