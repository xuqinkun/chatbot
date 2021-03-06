from model import *
from train import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    args = parse()
    model_save_path = args.model_path
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

    if model_save_path is None:
        model_save_path = DEFAULT_DATA_PATH
    model_save_dir = os.path.join(model_save_path, str(training_iteration), str(num_layers), str(num_hiddens))

    latest_training_model = None
    if os.path.exists(model_save_dir):
        file_list = os.listdir(model_save_dir)
        if len(file_list) > 0:
            numbers = [int(filename.split("_")[0]) for filename in file_list]
            max_num = sorted(numbers)[-1]
            latest_training_model = os.path.join(model_save_dir, "{}_{}.tar".format(max_num, MODEL_FILE_NAME))

    train(model, training_batches, lr, vocab, device, model_save_dir, latest_training_model)

