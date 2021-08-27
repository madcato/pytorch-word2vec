import os.path
import torch.nn as nn 
import torch
from word2vec.dataset import (DataLoader, Word2VecDataset)
from word2vec.model import (Word2VecModel)

words = DataLoader().read_data()

number_words = len(words)
print("Number of words: %i" % (number_words) )

dataset = Word2VecDataset(words)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

embedding_dims = 200
vocabulary_size = dataset.vocabulary_size

model = Word2VecModel(embedding_dims, vocabulary_size).to(device)
print(model)