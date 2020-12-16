import argparse
import os

from load import *
from model import *

DEFAULT_DATA_PATH = "data"

MODEL_FILE_NAME = 'epoch_model'

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


def train(model, data_iter, lr, num_epochs, vocab, device, model_save_dir, model_save_file=None):
    print("Training...")

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])

    model.apply(xavier_init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()

    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    start_epoch = 0

    if model_save_file and os.path.exists(model_save_file):
        checkpoint = torch.load(model_save_file)
        start_epoch = checkpoint['epoch']
        model.encoder.load_state_dict(checkpoint['en'])
        model.decoder.load_state_dict(checkpoint['de'])
        optimizer.load_state_dict(checkpoint['opt'])
    model.train()
    model.to(device)
    start = time.time()
    for epoch in range(start_epoch, num_epochs, 1):
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([vocab['<bos>']] * Y.shape[0],
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
        print(model)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save({
            'epoch': epoch + 1,
            'en': model.encoder.state_dict(),
            'de': model.decoder.state_dict(),
            'opt': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(model_save_dir, '{}_{}.tar'.format(epoch + 1, MODEL_FILE_NAME)))

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def predict(model, src_sentence, vocab, num_steps,
            device):
    """Predict sequences."""
    src_tokens = vocab[src_sentence.lower().split(' ')] + [
        vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Once the end-of-sequence token is predicted, the generation of
        # the output sequence is complete
        if pred == vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(vocab.to_tokens(output_seq))


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-mb', '--model_path', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Load the corpus file')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-ne', '--num_epoch', type=int, default=100, help='Train the model with n epochs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    model_save_path = args.model_path

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1

    batch_size, num_steps = 64, 200
    data_size = args.iteration
    lr, num_epochs, device = 0.005, args.num_epoch, d2l.try_gpu()

    train_iter, vocab = load_data("data/movie_subtitles.txt", data_size, num_steps, batch_size)
    encoder = Seq2SeqEncoder(len(vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(vocab), embed_size, num_hiddens, num_layers, dropout)
    model = d2l.EncoderDecoder(encoder, decoder)

    if model_save_path is None:
        model_save_path = DEFAULT_DATA_PATH
    model_save_dir = os.path.join(model_save_path, str(num_layers), str(num_hiddens))

    latest_training_model = None
    if os.path.exists(model_save_dir):
        file_list = os.listdir(model_save_dir)
        numbers = [int(filename.split("_")[0]) for filename in file_list]
        max_num = sorted(numbers)[-1]
        latest_training_model = os.path.join(model_save_dir, "{}_{}.tar".format(max_num, MODEL_FILE_NAME))

    train(model, train_iter, lr, num_epochs, vocab, device, model_save_dir, latest_training_model)

    while True:
        src_sentence = input(">")
        pred = predict(model, src_sentence, vocab, num_steps, device)
        print(pred)
