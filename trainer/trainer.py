import numpy as np
import torch
from tqdm import tqdm
from trainer import BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, model, data_loader, valid_data_loader,
                 criterion, optimizer, epochs, device, save_dir,
                 metric_ftns=None, lr_scheduler=None, resume_path=None):

        super().__init__(model, data_loader, valid_data_loader,
                         criterion, optimizer, epochs, device, save_dir, metric_ftns,
                         lr_scheduler, resume_path)

    def _train_epoch(self, epoch):

        self.model.train()

        train_loss = 0.0
        train_acc = 0.0

        for batch_idx, batch in tqdm(enumerate(self.data_loader)):

            text, text_lengths = batch.text
            label = batch.label

            self.optimizer.zero_grad()

            output = self.model(text, text_lengths).squeeze(1)
            loss = self.criterion(output, label)

            acc = self.metric_ftns(output, label)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        return train_loss/len(self.data_loader), train_acc/len(self.data_loader)

    def _valid_epoch(self, epoch):

        self.model.eval()

        valid_loss = 0.0
        valid_acc = 0.0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.valid_data_loader)):

                text, text_lengths = batch.text
                label = batch.label

                output = self.model(text, text_lengths).squeeze(1)
                loss = self.criterion(output, label)

                acc = self.metric_ftns(output, label)
                valid_loss += loss.item()
                valid_acc += acc.item()

        return valid_loss/len(self.valid_data_loader), valid_acc/len(self.valid_data_loader)
