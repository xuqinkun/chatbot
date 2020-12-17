from d2l import torch as d2l
from torch import nn

from load import *
from model import MaskedSoftmaxCELoss


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(model, training_batches, lr, vocab, device, model_save_dir, model_save_file=None):
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

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for epoch in range(start_epoch, len(training_batches), 1):
        batch = training_batches[epoch]
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
        bos = torch.tensor([vocab['<bos>']] * Y.shape[0],
                           device=device).reshape(-1, 1)
        dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
        Y_hat, _ = model(X, dec_input, X_valid_len)
        l = loss(Y_hat, Y, Y_valid_len)
        l.sum().backward()  # Make the loss scalar for `backward`
        d2l.grad_clipping(model, 1)
        optimizer.step()
        print("Progress:{%.2f}%% Total time: %.2f s" % (
        round((epoch + 1) * 100 / len(training_batches)), time.time() - start),
              end="\r")
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch + 1,
                'en': model.encoder.state_dict(),
                'de': model.decoder.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_dir, '{}_{}.tar'.format(epoch + 1, MODEL_FILE_NAME)))
    print(model)


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
