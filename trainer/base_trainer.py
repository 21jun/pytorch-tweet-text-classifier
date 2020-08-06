import torch
import numpy as np
from abc import abstractmethod
from tqdm import tqdm


class BaseTrainer:

    def __init__(self, model, data_loader, valid_data_loader,
                 criterion, optimizer, epochs, device, save_dir,
                 metric_ftns=None, lr_scheduler=None, resume_path=None):

        self.start_epoch = 1
        self.device = device
        self.model = model.to(device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir
        self.resume_path = resume_path

        if self.resume_path is not None:
            self._resume_checkpoint(self.resume_path)

    def train(self, do_valid=True, do_save=False):

        best_acc = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs + 1):

            print("epoch", "|", epoch)
            info = {}
            train_loss, train_acc = self._train_epoch(epoch)
            info['train_loss'] = train_loss
            info['train_acc'] = train_acc

            if do_valid:
                valid_loss, valid_acc = self._valid_epoch(epoch)
                info['valid_loss'] = valid_loss
                info['valid_acc'] = valid_acc

                if valid_acc > best_acc:
                    print("Best_acc updated : ", best_acc, "->", valid_acc)
                    best_acc = valid_acc

                    if do_save:
                        self._save_checkpoint(epoch)

            self._progress(info)

    @abstractmethod
    def _train_epoch(self, epoch):

        return NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):

        return NotImplementedError

    def _save_checkpoint(self, epoch):

        if self.save_dir == None:
            return
        print("Saving Checkpoint...")
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),  # for model
            'optimizer': self.optimizer.state_dict()  # for optimizer
        }
        fliename = str(self.save_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, fliename)

    def _resume_checkpoint(self, resume_path):

        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Model Load Complete")

    def _progress(self, info):
        print(info)
