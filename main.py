import os
from model import TransformerEncoder
from model import MaskedSoftmaxCELoss
from model import TransformerDecoder
import torch
import argparse
from d2l import torch as d2l
from torch import nn
from load import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = d2l.MLPAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        # Transpose outputs to (batch_size, seq_len, num_hiddens)
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_len

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs = []
        for x in X:
            # query shape: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context has same shape as query
            context = self.attention_cell(
                query, enc_outputs, enc_outputs, enc_valid_len)  # key=enc_outputs && value=enc_outputs
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x to (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_len]


def train(model, data_iter, lr, num_epochs, tgt_vocab, device):
    print("Training...")

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])

    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    model.train()
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for epoch in range(num_epochs):
        start = time.time()
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        print("Progress:{%.2f}%% Total time: %.2f s" % (round((epoch + 1) * 100 / num_epochs), time.time() - start),
              end="\r")
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def predict(model, src_sentence, src_vocab, tgt_vocab, num_steps,
            device):
    """Predict sequences."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Once the end-of-sequence token is predicted, the generation of
        # the output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    # parser.add_argument('-tr', '--train', help='Train the model with corpus')
    # parser.add_argument('-te', '--test', help='Test the saved model')
    # parser.add_argument('-l', '--load', help='Load the model and train')
    # parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    # parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    # parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    # parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    # parser.add_argument('-p', '--print', type=int, default=100, help='Print every p iterations')
    # parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    # parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    # parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    # parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    # parser.add_argument('-s', '--save', type=int, default=500, help='Save every s iterations')
    # parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    # parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout probability for rnn and dropout layers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 100
    data_size = args.iteration
    lr, num_epochs, device = 0.005, 1, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data("data/movie_subtitles.txt", data_size, num_steps, batch_size)
    encoder = d2l.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    model = d2l.EncoderDecoder(encoder, decoder)
    train(model, train_iter, lr, num_epochs, tgt_vocab, device)

    while True:
        src_sentence = input(">")
        pred = predict(model, src_sentence, src_vocab, tgt_vocab, num_steps, device)
        print(pred)
