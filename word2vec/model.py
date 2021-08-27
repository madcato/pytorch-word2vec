from torch import nn

class Word2VecModel(nn.Module):
    def __init__(self, embedding_dims, vocabulary_size):
        super(Word2VecModel, self).__init__()
        self.layer1 = nn.Linear(vocabulary_size, embedding_dims)
        self.layer2 = nn.Linear(embedding_dims, vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x
