import unicodedata
import re
import random

import numpy as np
SOS_token, EOS_token = 0, 1
MAX_LENGTH = 75

from word_embedding as embed
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.num_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII
    # refer to https://stackoverflow.com/a/518232/2809427
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize(sen):
    sen = unicodeToAscii(sen.lower().strip())
    sen = re.sub(r"([.!?])", r" \1", sen)
    sen = re.sub(r"[^a-zA-z.!?]+", r" ", sen)

    return sen


def read_from_text(q, a):
    print('Reading....')
    lines = open('./data/%s-%s.txt' % (q, a), encoding='utf-8').read().strip().split('\n')

    pairs = [[normalize(s) for s in l.split('\t')] for l in lines]

    questions=Lang(q)
    answers = Lang(a)

    return  questions,answers,pairs


def filter_pair(p):
    return len(p[0])<MAX_LENGTH and len(p[1]) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def perpare_dataset(q,a):
    input,output,pairs=read_from_text(q,a)
    print('Read {} sentence pairs'.format(len(pairs)))

    pairs=filter_pairs(pairs)

    for pair in pairs:
        input.add_sentence(pair[0])
        output.add_sentence(pair[1])
    print('Counted words: {}- {}\t{} - {}'.format(input.name, input.num_words,
                                                  output.name, output.num_words))

    return input, output, pairs

def prepareEmbMatrix(trained_vector_paths,vector_size,input_lang,output_size):
    input_emb_matrix=embed.get_embedding_matriz(input_lang.word2index,
                                                    trained_vector_paths[0], vector_size)
    output_emb_matrix = embed.get_embedding_matrix(output_lang.word2index,
                                                    trained_vector_paths[1], vector_size)
    SOS_token_vec = embed.initSpecialToken(vector_size, 0)  # SOS
    EOS_token_vec = embed.initSpecialToken(vector_size, 0)  # EOS

    for idx, tvec in enumerate((SOS_token_vec, EOS_token_vec)):
        input_emb_matrix[idx] = tvec
        output_emb_matrix[idx] = tvec
    return input_emb_matrix, output_emb_matrix

if __name__ == "__main__":
    '''
    The full process for preparing the data is:
        1. Read text file and split into lines, split lines into pairs
        2. Normalize text, filter by length and content
        3. Make word lists from sentences in pairs
    '''
    input_lang, output_lang, pairs = perpare_dataset('q', 'a')

    print(random.choice(pairs))

    trained_vector_paths=('cc.en.300.vec','cc.en.300.vec')

    input_emb_matrix,output_emb_matrix=prepareEmbMatrix(trained_vector_paths, 300, input_lang, output_lang)
    print('Embedding-matrix shape: {}, {}'.format(input_emb_matrix.shape, output_emb_matrix.shape))

    np.save('input_emb_matrix', input_emb_matrix)
    np.save('output_emb_matrix', output_emb_matrix)