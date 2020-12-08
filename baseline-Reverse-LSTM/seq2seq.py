import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, int(hidden_size / 2), bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedding = self.embedding(input).view(1, 1, -1)
        output = embedding

        output, hidden = self.lstm(output, hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1 * 2, 1, int(self.hidden_size / 2))


class BiLSTMEncoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=False, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden
