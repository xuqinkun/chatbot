import collections
import os
import random
import time

import torch


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
    max_len = 0
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
        max_len = max(len(parts[0]), len(parts[1]), max_len)
    return source, target, max_len


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [[vocab[l] for l in batch] for batch in lines]
    for batch in lines:
        for line in batch:
            line.append(vocab['<eos>'])
    array = torch.tensor([[truncate_pad(
        l, num_steps, vocab['<pad>']) for l in batch] for batch in lines])
    valid_len = []
    for batch in lines:
        temp = []
        for line in batch:
            temp.append(len(line))
        valid_len.append(temp)
    valid_len = torch.tensor(valid_len, dtype=torch.int32)
    return array, valid_len


def read_data(corpus, training_iteration, num_steps, batch_size):
    text = preprocess_nmt(read_text(corpus))

    vocab = Vocab(text.split(" "), min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    source, target, max_len = tokenize_nmt(text)
    pairs = [random_select(batch_size, source, target)
             for _ in range(training_iteration)]
    source = [pair[0] for pair in pairs]
    target = [pair[1] for pair in pairs]

    src_array, src_valid_len = build_array(source, vocab, num_steps)
    tgt_array, tgt_valid_len = build_array(target, vocab, num_steps)

    data_batches = []
    for i in range(len(src_array)):
        data_batches.append((src_array[i], src_valid_len[i], tgt_array[i], tgt_valid_len[i]))
    return data_batches, vocab


def random_select(batch_size, source, target):
    select_list = [i for i in range(batch_size)]
    random.shuffle(select_list)
    source = [source[i] for i in select_list]
    target = [target[i] for i in select_list]
    return source, target


def load_data(corpus_file, training_iteration, num_steps, batch_size):
    start = time.time()
    root_dir = corpus_file.split(os.sep)[0]
    corpus_save_dir = os.path.join(root_dir, str(training_iteration), str(num_steps), str(batch_size))
    vocab_save_file = os.path.join(root_dir, "vocab.tar")
    data_save_file = os.path.join(corpus_save_dir, "data.tar")
    try:
        vocab = torch.load(vocab_save_file)
        data_batches = torch.load(data_save_file)
    except FileNotFoundError:
        data_batches, vocab = read_data(corpus_file, training_iteration, num_steps, batch_size)
        if not os.path.exists(corpus_save_dir):
            os.makedirs(corpus_save_dir)
        torch.save(vocab, vocab_save_file)
        torch.save(data_batches, data_save_file)

    print("Load data: %.2f s" % (time.time() - start))
    return data_batches, vocab


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
