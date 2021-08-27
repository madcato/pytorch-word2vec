from torch import nn

class Word2VecModel(nn.Module):
    def __init__(self, embedding_dims, vocabulary_size):
        super(Word2VecModel, self).__init__()
        self.layer1 = nn.Linear(vocabulary_size, embedding_dims, bias=False)
        self.layer2 = nn.Linear(embedding_dims, vocabulary_size, bias=False)  # bias=False to not to learn additive bias
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax() produces positive losses, Softmax() negatives
                                             # dim=1 to apply to output. (dim=0 applies to batch dimension)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x
