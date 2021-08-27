import os.path
import torch.nn as nn 
import torch
from word2vec.dataset import (DataLoader, Word2VecDataset)
from word2vec.model import (Word2VecModel)
from word2vec.train import (train)

words = DataLoader().read_data()

number_words = len(words)
print("Number of words: %i" % (number_words) )

dataset = Word2VecDataset(words)

embedding_dims = 200
vocabulary_size = dataset.vocabulary_size

model = Word2VecModel(embedding_dims, vocabulary_size)
print(model)

train(dataset, model)