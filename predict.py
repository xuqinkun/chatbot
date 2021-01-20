# -*- coding: utf-8 -*-
from utils import *
from model import *
from load import *
from d2l import torch as d2l


def predict(model, src_sentence, vocab, num_steps, device):
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


if __name__ == '__main__':
    args = parse()
    model_save_file = args.model_path
    corpus_file = args.corpus

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1

    batch_size, num_steps = 64, 30
    training_iteration = args.iteration
    lr, num_epochs, device = 0.005, args.num_epoch, d2l.try_gpu()

    training_batches, vocab = load_data(corpus_file, training_iteration, num_steps, batch_size)
    encoder = Seq2SeqEncoder(len(vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(vocab), embed_size, num_hiddens, num_layers, dropout)
    model = d2l.EncoderDecoder(encoder, decoder)
    src_sentence = None

    if model_save_file and os.path.exists(model_save_file):
        checkpoint = torch.load(model_save_file)
        start_epoch = checkpoint['epoch']
        model.encoder.load_state_dict(checkpoint['en'])
        model.decoder.load_state_dict(checkpoint['de'])

    while src_sentence != "q":
        src_sentence = input(">")
        pred = predict(model, src_sentence, vocab, num_steps, device)
        print(pred)