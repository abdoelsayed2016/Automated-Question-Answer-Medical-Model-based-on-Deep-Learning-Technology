import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import argparse
import time, math, random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
from torch import optim

import data as dataloader

import seq2seq

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--num_iters', required=True, type=int)
    p.add_argument('--embedding_size', type=int, default=300)
    p.add_argument('--hidden_size', type=int, default=300)
    p.add_argument('--type',required=True, type=str)

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
          criterion,type='gru', max_length=dataloader.MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length, target_length = input_tensor.size(0), target_tensor.size(0)
    #print(type)
    if type=='gru':
        encoder_hidden = encoder.init_hidden().to(device)
    else:
        #print("asdadsadadasaaaaaaaaaaaaaaaaa")
        encoder_hidden = (encoder.init_hidden().to(device), encoder.init_hidden().to(device))

    print(max_length,encoder.hidden_size)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input=torch.tensor([[dataloader.SOS_token]]).to(device)

    decoder_hidden=encoder_hidden

    use_teacher_forcing=True if random.random() < teacher_focing_ratio else False

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)

            loss+=criterion(decoder_output,target_tensor[di])

            decoder_input=target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)

            topv,topi=decoder_output.topk(1)

            decoder_input=topi.squeeze().detach()
            loss+=criterion(decoder_output,target_tensor[di])

            if decoder_input.item()==dataloader.EOS_token:
                break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return  loss.item()/target_length

def trainiters(pairs,encoder,decoder,n_iters,
               train_pairs_seed=0, print_every=1000, plot_every=1000, learning_rate=.01,type='gru'):
    start=time.time()
    plot_losses=[]
    print_loss_total,plot_loss_total=0,0

    train_pairs=[tensorFromPair(random.choice(pairs)) for i in range(n_iters)]

    encoder_optimizer=optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer=optim.SGD(decoder.parameters(),lr=learning_rate)

    criterion=nn.NLLLoss()

    for iter in range(1,n_iters+1):
        pair=train_pairs[iter-1]

        input_tensor,target_tensor=pair[0],pair[1]

        loss=train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,type=type)

        print_loss_total+=loss
        plot_loss_total+=loss

        if iter %print_every==0:
            print_loss_avg=print_loss_total/print_every
            print_loss_total=0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every:
            plot_loss_avg=plot_loss_total/print_every

            plot_losses.append(plot_loss_avg)
            plot_loss_total=0
    showPlot(plot_losses)
    if type=='gru':

        plt.savefig('baseline-GRU-loss')
    else:
        plt.savefig('baseline-LSTM-loss')
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')




if __name__ == "__main__":
    config = argparser()
    teacher_focing_ratio = 0.5

    input_lang, output_lang, pairs = dataloader.perpare_dataset('q', 'a')

    encoder = seq2seq.GRUEncodeer(input_size=input_lang.num_words, embedding_size=config.embedding_size,
                                  hidden_size=config.hidden_size,type=config.type).to(device)

    decoder = seq2seq.GRUDecoder(output_size=output_lang.num_words,
                                 embedding_size=config.embedding_size,
                                 hidden_size=config.hidden_size,type=config.type).to(device)


    print(encoder,decoder)
    if os.path.isfile('./encoder.pth'):

        encoder.load_state_dict(torch.load('encoder.pth'))
        decoder.load_state_dict(torch.load('decoder.pth'))


    trainiters(pairs, encoder, decoder, config.num_iters,type=config.type)
