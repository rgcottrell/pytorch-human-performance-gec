import numpy as np

from fairseq.data.indexed_dataset import IndexedDataset
from fairseq.tokenizer import Tokenizer

class IndexedRawStringDataset(IndexedDataset):
    """Takes a string as input and binarizes it in memory at instantiation.
    Original string is also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, line, dictionary):
        self.lines.append(line.strip('\n'))
        tokens = Tokenizer.tokenize(
            line, dictionary, add_if_not_exist=False,
            append_eos=self.append_eos, reverse_order=self.reverse_order,
        ).long()
        self.tokens_list.append(tokens)
        self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)