from d2l import torch as d2l
from torch import nn

from load import *
from model import MaskedSoftmaxCELoss
from utils import *


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
