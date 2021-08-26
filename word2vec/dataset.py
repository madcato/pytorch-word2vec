import collections
import os.path
import numpy
from torch.utils.data import Dataset
import torch
from torchtext.data.datasets_utils import (_download_extract_validate)
from tqdm import tqdm

class DataLoader:
    def __init__(self):
        self.data_path = "."
        self.file_name = "text8"
        self.download_text8_file_in(self.data_path, self.file_name)
    
    def download_text8_file_in(self, data_path, file_name):
        url = "http://mattmahoney.net/dc/text8.zip"
        _download_extract_validate(data_path, 
                                   url, 
                                   "a6640522afe85d1963ad56c05b0ede0a0c000dddc9671758a6cc09b7a38e5232",
                                   os.path.join(data_path, "text8.zip"), 
                                   os.path.join(data_path, file_name), 
                                   "6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7")

    def read_data(self):
        """Read file word by word and return it in an array."""
        file_name = self.data_path + "/" + self.file_name
        data = []
        with open(file_name,'r') as file:
            for line in file:
                for word in line.split():
                    data.append(word)
        return data

class Word2VecDataset(Dataset):
    """Store data in memory"""
    def __init__(self, word_array, window_size=1, max_size_vocabulary=4000):
        pre_saved_data_file = "text8.pairs.npy"
        vocabulary_file = "text8_vocabulary.npy"
        self.data = []
        self.vocabulary_size = 0
        self.vocabulary = []
        self.window_size = window_size
        self.data = self.load_data(pre_saved_data_file, vocabulary_file)
        if len(self.data) == 0:
            self.data = self.prepare_data(word_array, max_size_vocabulary)
            self.save_data(pre_saved_data_file, vocabulary_file, self.data)
    
    def __len__(self):
        """return the size of the data"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return an item to train. Each item the imput and the output to train"""
        sample = self.data[idx]
        return self.get_input_layer(sample[0]), self.get_input_layer(sample[1])

    def prepare_data(self, word_array, max_size_vocabulary):
        """Process raw inputs into a dataset."""
        print("Preparing dictionary")
        frequencies = dict()
        for word in tqdm(word_array):
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1
        
        self.vocabulary = self.filter_infrequent_words(frequencies, max_size_vocabulary)

        self.vocabulary_size = len(self.vocabulary)

        frequencies = None

        word2idx = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        idx2word = {idx: w for (idx, w) in enumerate(self.vocabulary)}

        pairs = self.generate_pairs(word_array, word2idx, idx2word)

        return pairs

    def generate_pairs(self, word_array, word2idx, idx2word):
        print("Generating pairs: ")
        idx_pairs = numpy.empty((0, 2), dtype=int)
        array_size = len(word_array)
        # for each word
        for idx, word in tqdm(enumerate(word_array)):
            ## Temporal fix BEGIN
            if len(idx_pairs) >= 40000:
                break
            ## Temporal fix END
            if word not in self.vocabulary:
                continue
            
            for w in range(-self.window_size, self.window_size + 1):
                context_word_pos = idx + w
                if context_word_pos < 0 or context_word_pos >= array_size or idx == context_word_pos:
                    continue
                context_word = word_array[context_word_pos]
                if context_word not in self.vocabulary:
                    continue
                idx_pairs = numpy.append(idx_pairs, [[word2idx[word], word2idx[context_word]]], axis=0)

        return idx_pairs

    def prepare_samples(self, pairs):
        print("Preparing samples: ")
        samples = []
        for source, target in tqdm(pairs):
            x = self.get_input_layer(source)
            y = self.get_input_layer(target)
            samples.append([x, y])

        return samples

    def get_input_layer(self, word_idx):
        tensor = torch.zeros(self.vocabulary_size).float()
        tensor[word_idx] = 1.0
        return tensor

    def filter_infrequent_words(self, frequencies, max_size_vocabulary):
        print("Removing infrequent words")
        temp_vec = []
        for word, freq in tqdm(frequencies.items()):
            temp_vec.append(freq)
        
        temp_vec.sort(reverse=True)

        min_freq = temp_vec[max_size_vocabulary - 1]

        print("Building vocabulary")
        new_vocabulary = []
        for word, freq in tqdm(frequencies.items()):
            if freq > min_freq:
                new_vocabulary.append(word)
        
        return new_vocabulary

    def load_data(self, pre_saved_data_file, vocabulary_file):
        """try to load pregenerated data"""
        if os.path.isfile(pre_saved_data_file):
            if os.path.isfile(vocabulary_file) == False:
                print("Vacabulary file required or regenerate saved data")
                exit()
            self.vocabulary = list(numpy.load(vocabulary_file))
            self.vocabulary_size = len(self.vocabulary)
            return numpy.load(pre_saved_data_file)
        else:
            return []

    def save_data(self, pre_saved_data_file, vocabulary_file, data):
        """save pregenerated data"""
        numpy.save(vocabulary_file, numpy.array(self.vocabulary))
        numpy.save(pre_saved_data_file, data)


