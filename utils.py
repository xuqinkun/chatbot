# -*- coding: utf-8 -*-
import argparse
import torch

DEFAULT_DATA_PATH = "data"
MODEL_FILE_NAME = "model_backup"


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-mb', '--model_path', help='Load the model and train')
    parser.add_argument('-c', '--corpus', required=True, help='Load the corpus file')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-ne', '--num_epoch', type=int, default=100, help='Train the model with n epochs')
    args = parser.parse_args()
    return args


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')