import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from os.path import dirname, abspath, join, exists

BASE_DIR = dirname(abspath(__file__))


class BaseTrainer:

    def __init__(self, model, train_dataloader, val_dataloader, loss_function, optimizer,
                 device=None, print_every=1, save_every=100, logger=None, save_name=None):

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

        self.print_every = print_every
        self.save_every = save_every
        self.save_name = save_name

        self.epoch = 0
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        # self.train_epoch_accuracies = []
        # self.val_epoch_accuracies = []

        self.logger = logger

        self.start_time = datetime.now()

    def train(self, epoch):
        self.model.train()

        train_batch_losses = []
        # train_batch_losses, train_batch_metrics = [], []
        for train_targets, train_inputs in tqdm(self.train_dataloader):
            train_inputs, train_targets = train_inputs.to(self.device), train_targets.to(self.device)

            train_outputs = self.model(train_inputs)

            # TODO: Mask loss
            vocabulary_size = train_outputs.size(-1)
            train_outputs_flat = train_outputs.view(-1, vocabulary_size)
            train_targets_flat = train_targets.view(-1)
            train_batch_loss = self.loss_function(train_outputs_flat, train_targets_flat)

            self.optimizer.zero_grad()
            train_batch_loss.backward()
            self.optimizer.step()

            train_batch_losses.append(train_batch_loss.item())

            # train_batch_metric = self._calculate_metric(train_outputs, train_targets)
            # train_batch_metrics.append(train_batch_metric)

            if self.epoch == 0:  # for debugging
                break

        train_data_size = len(self.train_dataloader.dataset)
        train_epoch_loss = sum(train_batch_losses) / train_data_size
        # train_epoch_accuracy = sum(train_batch_metrics).item() / train_data_size

        if epoch % self.print_every == 0:
            self.model.eval()
            val_batch_losses = []
            # val_batch_losses, val_batch_metrics = [], []
            for val_targets, val_inputs in self.val_dataloader:
                val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)

                val_outputs = self.model(val_inputs)

                vocabulary_size = val_outputs.size(-1)
                val_outputs_flat = val_outputs.view(-1, vocabulary_size)
                val_targets_flat = val_targets.view(-1)
                val_batch_loss = self.loss_function(val_outputs_flat, val_targets_flat)

                val_batch_losses.append(val_batch_loss.item())

                # val_batch_metric = self._count_corrects(val_outputs, val_targets)
                # val_batch_metrics.append(val_batch_metric)

            val_data_size = len(self.val_dataloader.dataset)
            val_epoch_loss = sum(val_batch_losses) / val_data_size
            # val_epoch_accuracy = sum(val_batch_metrics).item() / val_data_size
        else:
            val_epoch_loss = 0
            # val_epoch_accuracy = None

        return train_epoch_loss, val_epoch_loss  # , train_epoch_accuracy, val_epoch_accuracy

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            train_epoch_loss, val_epoch_loss = self.train(epoch)  # , train_epoch_accuracy, val_epoch_accuracy

            self.train_epoch_losses.append(train_epoch_loss)
            self.val_epoch_losses.append(val_epoch_loss)
            # self.train_epoch_accuracies.append(train_epoch_accuracy)
            # self.val_epoch_accuracies.append(val_epoch_accuracy)

            train_epoch_perplexity = self._calculate_perplexity(train_epoch_loss)
            val_epoch_perplexity = self._calculate_perplexity(val_epoch_loss)

            if epoch % self.print_every == 0 and self.logger:
                self._leave_log(train_epoch_loss, val_epoch_loss, train_epoch_perplexity,
                                val_epoch_perplexity, progress=epoch / epochs)

            if epoch % self.save_every == 0:
                if self.logger:
                    self.logger.info("Saving the model...")
                self._save_model()

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed)

    def _leave_log(self, train_epoch_loss, val_epoch_loss, train_epoch_perplexity, val_epoch_perplexity, progress):
        base_message = ("Epoch: {epoch:<3d} "
                        "Progress: {progress:<.1%} ({elapsed}) "
                        "Train Loss: {train_loss:<.6} "
                        "Val Loss: {val_loss:<.6} "
                        "Train Perplexity: {train_perplexity:<.3} "
                        "Val Perplexity: {val_perplexity:<.3} "
                        "Learning rate: {learning_rate:<.4} "
                        )
        current_lr = self.optimizer.param_groups[0]['lr']
        message = base_message.format(epoch=self.epoch,
                                      progress=progress,
                                      train_loss=train_epoch_loss,
                                      val_loss=val_epoch_loss,
                                      train_perplexity=train_epoch_perplexity,
                                      val_perplexity=val_epoch_perplexity,
                                      learning_rate=current_lr,
                                      elapsed=self._elapsed_time()
                                      )
        self.logger.info(message)

    @staticmethod
    def _calculate_perplexity(loss):
        return np.exp(loss)

    def _save_model(self):
        if self.save_name is None:
            self.save_name = self.model.__class__.__name__

        checkpoint_dir = join(BASE_DIR, 'parameters', self.save_name)
        if not exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        base_filename = '{model_name}-{start_time}-{epoch}.pth'
        checkpoint_filename = base_filename.format(model_name=self.save_name,
                                                   start_time=self.start_time,
                                                   epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)

        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if min(self.val_epoch_losses) == self.val_epoch_losses[-1]:  # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath


# TODO : Make sane collate_fn
def collate_fn(batch):
    """merges a list of samples to form a mini-batch."""

    text_lengths = [sentence.size(0) for sentiment_index, sentence in batch]
    longest_length = max(text_lengths)

    reviews_padded = [torch.cat([torch.zeros(longest_length - sentence.size(0), 128), sentence], dim=0) for
                      sentiment_index, sentence in
                      batch]
    labels = [sentiment_index for sentiment_index, sentence in batch]

    reviews_tensor = torch.stack(reviews_padded)
    labels_tensor = torch.stack(labels)  # classification
    return labels_tensor, reviews_tensor