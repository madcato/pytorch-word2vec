# pytorch-word2vec

word2vec implementation using PyTroch

## Links
- [word2vec](https://code.google.com/archive/p/word2vec/)
- [PyTroch Documentation](https://pytorch.org/docs/stable/index.html)
- [Wikipedia: word2vec](https://en.wikipedia.org/wiki/Word2vec)
- [Tutorial word2vec](https://www.tensorflow.org/tutorials/text/word2vec)
- [Another tutorial word2vec](https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/)
- [Word2Vec Explained](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/)
- [Word2Vec Explained (another)](https://towardsdatascience.com/word2vec-explained-49c52b4ccb71)
- [PyTorch Loss Functions: The Ultimate Guide](https://neptune.ai/blog/pytorch-loss-functions)
- [Create your Mini-Word-Embedding from Scratch using Pytorch](https://deepscopy.com/Create_your_Mini_Word_Embedding_from_Scratch_using_Pytorch)

### Data parallelism
- [Docs >  torch.nn > DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)
- [PyTroch tutorial: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
- [PyTroch tutorial: Getting Started With Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## PyTorch Tutotials
- [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Datasets & Datloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Build the neural network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

## requiremnets
- pytorch
- torchtext
- tqdm

## Run

    $ python3 run.py

The multi-gpu code runs only with three GPU. 

To run only in one GPU, deactivate the rest by executing:

    $ export CUDA_VISIBLE_DEVICES="0"
