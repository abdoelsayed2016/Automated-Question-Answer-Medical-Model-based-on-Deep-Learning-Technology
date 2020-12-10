import numpy as  np


def word2vec(trained_vector_path,vector_size):
    fasttext_path='../fasttext/'+trained_vector_path.format(vector_size)
    word2vec_dict={}

    with open(fasttext_path,'r',encoding='utf-8') as fh:
        for line in (fh):
            array=line.lstrip().rstrip().split(" ")
            word2vec_dict[array[0]]=list(map(float,array[1:]))

    return word2vec_dict

def initSpecialToken(vector_size,seed):
    np.random.seed(seed)
    return np.random.rand(vector_size)

def get_embedding_matriz(word2index,trained_vector_path,vector_size):
    trained_word_vec=word2vec(trained_vector_path,vector_size)


    embedding_matrix=np.zeros(len(word2index),vector_size)

    for word,idx in word2index.items():
        word_vec=trained_word_vec.get(word)
        if word_vec is not None:
            embedding_matrix[idx]=word_vec

    return embedding_matrix

if __name__=="__main__":
    pass
