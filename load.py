import collections
import os
import random
import time

import torch
from torch.utils import data


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def count_corpus(tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def read_text(corpus):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    # import gzip
    lines = [x.strip() for x in content]
    it = iter(lines)
    text = []
    for x in it:
        text.append("\t".join([x, next(it)]))
    return "\n".join(text)


def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = []
    for line in lines:
        if len(line) > num_steps:
            print("Large sentence: %d" % len(line))
        valid_len.append(len(line))
    valid_len = torch.tensor(valid_len, dtype=torch.int32)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def read_data(corpus, data_size, num_steps, batch_size):
    text = preprocess_nmt(read_text(corpus))

    vocab = Vocab(text.split(" "), min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    source, target = tokenize_nmt(text)
    source, target = random_select(data_size, source, target)
    src_array, src_valid_len = build_array(source, vocab, num_steps)
    tgt_array, tgt_valid_len = build_array(target, vocab, num_steps)

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, vocab


def random_select(data_size, source, target):
    select_list = [i for i in range(data_size)]
    random.shuffle(select_list)
    source = [source[i] for i in select_list]
    target = [target[i] for i in select_list]
    return source, target


def load_data(corpus_file, data_size, num_steps, batch_size):
    start = time.time()
    root_dir = corpus_file.split(os.sep)[0]
    corpus_save_dir = os.path.join(root_dir, str(data_size), str(num_steps), str(batch_size))
    vocab_save_file = os.path.join(root_dir, "vocab.tar")
    data_save_file = os.path.join(corpus_save_dir, "data.tar")
    try:
        vocab = torch.load(vocab_save_file)
        data_iter = torch.load(data_save_file)
        # src_valid_len = torch.load(corpus_processed)["src_valid_len"]
        # tgt_array = torch.load(corpus_processed)["tgt_array"]
        # tgt_valid_len = torch.load(corpus_processed)["tgt_valid_len"]
    except FileNotFoundError:
        data_iter, vocab = read_data(corpus_file, data_size, num_steps, batch_size)
        if not os.path.exists(corpus_save_dir):
            os.makedirs(corpus_save_dir)
        torch.save(vocab, vocab_save_file)
        torch.save(data_iter, data_save_file)

    print("Load data: %.2f s" % (time.time() - start))
    return data_iter, vocab


def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


if __name__ == '__main__':
    print(read_text("data/movie_subtitles.txt"))
