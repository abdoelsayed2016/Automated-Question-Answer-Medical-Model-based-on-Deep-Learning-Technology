import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncodeer(nn.module):
    def __init__(self,input_size,embedding_size,hidden_size):


        super(GRUEncodeer,self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size


        self.embedding=nn.Embedding(input_size,embedding_size)
        self.gru=nn.GRU(embedding_size,hidden_size,bidirectional=False,batch_first=True)

    def forward(self,input,hidden):

        embedded=self.embedding(input).view(1,1,-1)
        output=embedded
        output,hidden=self.gru(output,hidden)
        return  output,hidden
    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_size)

