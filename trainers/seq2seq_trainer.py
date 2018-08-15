import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from os.path import dirname, abspath, join, exists

BASE_DIR = dirname(dirname(abspath(__file__)))
PAD_INDEX = 0


class Seq2SeqTrainer:

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

        self.logger = logger

        self.start_time = datetime.now()

    def run_epoch(self, dataloader, mode='train'):

        batch_losses = []
        batch_token_counts = []
        for sources, inputs, targets, source_lengths, target_lengths in tqdm(dataloader):
            sources = sources.to(self.device)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(sources, inputs)

            vocabulary_size = outputs.size(-1)
            outputs_flat = outputs.view(-1, vocabulary_size)
            targets_flat = targets.view(-1)
            batch_loss = self.loss_function(outputs_flat, targets_flat)
            loss_mask = self.sequence_mask(target_lengths)
            batch_loss_masked = batch_loss.masked_fill(loss_mask, 0)
            batch_loss_summed = batch_loss_masked.sum()

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss_summed.backward()
                self.optimizer.step()

            batch_losses.append(batch_loss_summed.item())
            batch_token_count = self.model.inputs_mask.sum().item()
            batch_token_counts.append(batch_token_count)

            if self.epoch == 0:  # for debugging
                break

        token_counts = sum(batch_token_counts)
        epoch_loss_per_token = sum(batch_losses) / token_counts

        return epoch_loss_per_token

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.model.train()
            train_epoch_loss_per_token = self.run_epoch(self.train_dataloader, mode='train')

            self.model.eval()
            val_epoch_loss_per_token = self.run_epoch(self.val_dataloader, mode='val')

            self.train_epoch_losses.append(train_epoch_loss_per_token)
            self.val_epoch_losses.append(val_epoch_loss_per_token)

            if epoch % self.print_every == 0 and self.logger:
                train_epoch_perplexity = self._calculate_perplexity(train_epoch_loss_per_token)
                val_epoch_perplexity = self._calculate_perplexity(val_epoch_loss_per_token)

                self._leave_log(train_epoch_loss_per_token, val_epoch_loss_per_token,
                                train_epoch_perplexity, val_epoch_perplexity, progress=epoch / epochs)

            if epoch % self.save_every == 0:
                self._save_model()

    @staticmethod
    def sequence_mask(lengths, max_length=None):
        # lengths: (batch_size, )
        if not max_length:
            max_length = lengths.max()  # or predefined max_len
        batch_size = lengths.size(0)
        lengths_broadcastable = lengths.unsqueeze(1)
        mask = torch.arange(0, max_length).type_as(lengths).repeat(batch_size, 1) >= lengths_broadcastable
        # mask: (batch_size, seq_length)
        return mask.view(-1)

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

        base_filename = '{save_name}_{epoch}_{start_time}.pth'
        checkpoint_filename = base_filename.format(save_name=self.save_name,
                                                   start_time=self.start_time.strftime("%Y-%m-%d-%H:%M:%S"),
                                                   epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)

        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if min(self.val_epoch_losses) == self.val_epoch_losses[-1]:  # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))


def seq2seq_collate_fn(batch):
    """merges a list of samples to form a mini-batch.

    Args:
        batch : tuple of inputs and targets. For example,
        ([1, 164, 109, 253, 66, 484, 561, 76, 528, 279, 458],
        [164, 109, 253, 66, 484, 561, 76, 528, 279, 458, 1])
    """
    batch = [(sum(source, []), target) for source, target in batch]
    sorted_batch = sorted(batch, key=lambda source_target: len(source_target[0]), reverse=True)
    source_lengths = [len(source) for source, _ in sorted_batch]
    target_lengths = [len(target) for _, target in sorted_batch]

    longest_source_length = source_lengths[0]
    longest_target_length = max(target_lengths)

    sources_padded = [source + [PAD_INDEX] * (longest_source_length - len(source)) for source, target in batch]
    targets_padded = [target + [PAD_INDEX] * (longest_target_length - len(target)) for source, target in batch]

    sources_tensor = torch.tensor(sources_padded)
    targets_tensor = torch.tensor(targets_padded)

    inputs_tensor = targets_tensor[:, :-1].contiguous()
    targets_tensor = targets_tensor[:, 1:].contiguous()

    source_lengths_tensor = torch.tensor(source_lengths)
    target_lengths_tensor = torch.tensor(target_lengths) - 1  # - 1 for inputs / targets split

    return sources_tensor, inputs_tensor, targets_tensor, source_lengths_tensor, target_lengths_tensor
