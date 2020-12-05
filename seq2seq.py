import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncodeer(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,type):


        super(GRUEncodeer,self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.type=type

        self.embedding=nn.Embedding(input_size,embedding_size)
        if type=='gru':

            self.gru=nn.GRU(embedding_size,hidden_size,bidirectional=False,batch_first=True)
        else:
            self.lstm=nn.LSTM(embedding_size,hidden_size,bidirectional=False,batch_first=True)

    def forward(self,input,hidden):

        embedded=self.embedding(input).view(1,1,-1)
        output=embedded
        if self.type=='gru':
            output,hidden=self.gru(output,hidden)
        else:
            output, hidden = self.lstm(output, hidden)
        return  output,hidden
    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_size)

class GRUDecoder(nn.Module):
    def __init__(self,output_size,embedding_size,hidden_size,type):
        super(GRUDecoder,self).__init__()
        self.output_size=output_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.type=type
        self.embedding=nn.Embedding(output_size,embedding_size)
        if type=='gru':
            self.gru=nn.GRU(embedding_size,hidden_size,bidirectional=False,batch_first=True)
        else:
            self.lstm=nn.LSTM(embedding_size,hidden_size,bidirectional=False,batch_first=True)

        self.out=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)
    def forward(self,input,hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        if self.type=='gru':

            output, hidden = self.gru(output, hidden)
        else:
            output, hidden = self.lstm(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden