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

class Multi3GPUWord2VecModel(nn.Module):
    def __init__(self, embedding_dims, vocabulary_size, dev0, dev1, dev2):
        super(Multi3GPUWord2VecModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2
        self.layer1 = nn.Linear(vocabulary_size, embedding_dims, bias=False).to(self.dev0)
        self.layer2 = nn.Linear(embedding_dims, vocabulary_size, bias=False).to(self.dev1)  # bias=False to not to learn additive bias
        self.softmax = nn.LogSoftmax(dim=1).to(self.dev2)  # LogSoftmax() produces positive losses, Softmax() negatives
                                             # dim=1 to apply to output. (dim=0 applies to batch dimension)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.layer1(x).to(self.dev1)
        x = self.layer2(x).to(self.dev2)
        x = self.softmax(x)
        return x
