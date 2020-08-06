from dataset.preprocess import preprocess_text
from torchtext import data
from dataset.dataset import DataFrameDataset
from model.rnn_model import LSTM_net
from model.metric import binary_accuracy
from trainer.trainer import Trainer
from pathlib import Path

import torch


MAX_VOCAB_SIZE = 20000
PRETRAINED = 'glove.6B.200d'
BATCH_SIZE = 1024

EPOCHS = 20
LEARNING_RATE = 0.005

EMBEDDING_DIM = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.1

PATH = Path("outputs")

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = preprocess_text('data')
    train_df, valid_df, test_df = df['train'], df['valid'], df['test']

    TEXT = data.Field(tokenize='spacy', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [('text', TEXT), ('label', LABEL)]

    train_ds, val_ds, test_ds = DataFrameDataset.splits(
        fields, train_df=train_df, val_df=valid_df, test_df=test_df)

    TEXT.build_vocab(train_ds,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=PRETRAINED,
                     unk_init=torch.Tensor.zero_)
    LABEL.build_vocab(train_ds)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        device=device)

    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # padding
    pretrained_embeddings = TEXT.vocab.vectors

    model = LSTM_net(INPUT_DIM,
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     N_LAYERS,
                     BIDIRECTIONAL,
                     DROPOUT,
                     PAD_IDX,
                     pretrained_embeddings)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(model, train_iterator, valid_iterator,
                      criterion, optimizer, EPOCHS, device, PATH, binary_accuracy)

    trainer.train(do_valid=True, do_save=True)
