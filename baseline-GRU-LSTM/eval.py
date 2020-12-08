import argparse

import time

import numpy as np
import torch
import  data
import seq2seq
import main
from sklearn.model_selection import train_test_split

from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    p=argparse.ArgumentParser()

    p.add_argument('--encoder',
                   required=True)

    p.add_argument('--decoder',
                   required=True)

    p.add_argument('--embedding_size',
                   type=int,
                   default=300)

    p.add_argument('--hidden_size',
                   type=int,
                   default=300)
    p.add_argument('--type', required=True, type=str)
    config=p.parse_args()

    return config

def predict_sentence(pair,output):
    print('Question: {}\nAnswer: {}'.format(pair[0],pair[1]))
    print('Predict Answer:{}'.format(output),end='\n\n')


def evaluate(sentence,encoder,decoder,max_length=data.MAX_LENGTH,type='gru'):
    with torch.no_grad():
        input_tensor=main.tensorFromSentences(input_lang,sentence)

        input_length=input_tensor.size(0)

        if type == 'gru':
            encoder_hidden = encoder.init_hidden().to(device)
        else:
            # print("asdadsadadasaaaaaaaaaaaaaaaaa")
            encoder_hidden = (encoder.init_hidden().to(device), encoder.init_hidden().to(device))

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # |encoder_output| = (batch_size, sequence_length, num_directions*hidden_size)
            # |encoder_hidden| = (num_layers*num_directions, batch_size, hidden_size)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[data.SOS_token]]).to(device)
        # |decoder_input| = (1, 1)
        decoder_hidden = encoder_hidden
        # |decoder_hidden|= (num_layers*num_directions, batch_size, hidden_size)

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (num_layers*num_directions, batch_size, hidden_size)

            topv, topi = decoder_output.data.topk(1)  # top-1 value, index
            # |topv|, |topi| = (1, 1)

            if topi.item() == data.EOS_token:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words


def evaluateiters(pairs, encoder, decoder, train_pairs_seed=0):
    start = time.time()
    cc = SmoothingFunction()
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    # |test_pairs| = (n_pairs, 2, sentence_length, 1) # eng, fra

    scores = []
    for pi, pair in enumerate(test_pairs):
        output_words = evaluate(pair[0], encoder, decoder)
        output_sentence = ' '.join(output_words)

        # for print
        predict_sentence(pair, output_sentence)

        # for nltk.bleu
        ref = pair[1].split()
        hyp = output_words
        scores.append(sentence_bleu([ref], hyp, smoothing_function=cc.method3) * 100.)

    print('BLEU: {:.4}'.format(sum(scores) / len(test_pairs)))


if __name__ == "__main__":
    '''
    Evaluation is mostly the same as training,
    but there are no targets so we simply feed the decoder's predictions back to itself for each step.
    Every time it predicts a word, we add it to the output string,
    and if it predicts the EOS token we stop there.
    '''
    config = argparser()

    input_lang, output_lang, pairs = data.perpare_dataset('q', 'a')

    encoder = seq2seq.GRUEncoder(input_size=input_lang.n_words,
                                 embedding_size=config.embedding_size,
                                 hidden_size=config.hidden_size
                                 ).to(device)

    decoder = seq2seq.GRUDecoder(output_size=output_lang.n_words,
                                 embedding_size=config.embedding_size,
                                 hidden_size=config.hidden_size
                                 ).to(device)

    encoder.load_state_dict(torch.load(config.encoder))
    encoder.eval()
    decoder.load_state_dict(torch.load(config.decoder))
    decoder.eval()

    evaluateiters(pairs, encoder, decoder)