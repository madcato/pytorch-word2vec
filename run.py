import os.path
import torch.nn as nn 
import torch
from word2vec.dataset import (DataLoader, Word2VecDataset)

words = DataLoader().read_data()

number_words = len(words)
print("Number of words: %i" % (number_words) )

dataset = Word2VecDataset(words)
print(len(dataset))
print(dataset[8])

