import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMEncoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,embedding_matrix):

        super(BiLSTMEncoder,self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size

        self.hidden_size=hidden_size

        self.embedding=nn.Embedding(input_size,embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm=nn.LSTM(embedding_size,int(hidden_size/2),bidirectional=True,batch_first=True)
    def forward(self, input,hidden):
        embedded =self.embedding.view(1,1,-1)
        output=embedded

        output,hidden=self.lstm(output,hidden)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1*2,1,int(self.hidden_size/2))

