import os.path
import torch.nn as nn 
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from word2vec.dataset import (DataLoader, Word2VecDataset)
from word2vec.model import (Word2VecModel, Multi3GPUWord2VecModel)
from word2vec.train import (train)
import word2vec.utils as pp

words = DataLoader().read_data()

number_words = len(words)
print("Number of words: %i" % (number_words) )

dataset = Word2VecDataset(words)

embedding_dims = 200
vocabulary_size = dataset.vocabulary_size


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def run_model_parallel(rank, world_size):
    pp.setup(rank, world_size)
    dev0 = 'cuda:0'
    dev1 = 'cuda:1'
    dev2 = 'cuda:2'
    model = Multi3GPUWord2VecModel(embedding_dims, vocabulary_size, dev0, dev1, dev2) 
    model = DDP(model)  #, device_ids=[0, 1, 2])
    print(model)
    device = dev0
    train(dataset, model, device)
    pp.cleanup()

if __name__ == "__main__":
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        world_size = 3  # Three gpus
        pp.run(run_model_parallel, world_size)
    else:    
        model = Word2VecModel(embedding_dims, vocabulary_size)    
        model = model.to(device)  # it's hudge important to move model to device, before creating optimizer
        train(dataset, model, device)    


