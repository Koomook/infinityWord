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

    def train(self, epoch):
        self.model.train()

        train_batch_losses = []
        train_batch_token_counts = []
        for train_sources, train_inputs, train_targets, train_source_lengths, train_target_lengths in tqdm(self.train_dataloader):
            train_sources = train_sources.to(self.device)
            train_inputs = train_inputs.to(self.device)
            train_targets = train_targets.to(self.device)

            train_decoder_outputs, train_decoder_state, train_attentions = self.model(train_sources, train_inputs, train_source_lengths)

            vocabulary_size = train_decoder_outputs.size(-1)
            train_outputs_flat = train_decoder_outputs.view(-1, vocabulary_size)
            train_targets_flat = train_targets.view(-1)
            train_batch_loss = self.loss_function(train_outputs_flat, train_targets_flat)

            self.optimizer.zero_grad()
            train_batch_loss.backward()
            self.optimizer.step()

            train_batch_losses.append(train_batch_loss.item())
            train_batch_token_count = train_targets_flat.size(0)
            train_batch_token_counts.append(train_batch_token_count)

            if self.epoch == 0:  # for debugging
                break

        train_token_counts = sum(train_batch_token_counts)
        train_epoch_loss_per_token = sum(train_batch_losses) / train_token_counts

        if epoch % self.print_every == 0:
            self.model.eval()
            val_batch_losses = []
            val_batch_token_counts = []
            for val_sources, val_inputs, val_targets, val_source_lengths, val_target_lengths in self.val_dataloader:
                val_sources = val_sources.to(self.device)
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)

                val_decoder_outputs, val_decoder_state, val_attentions = self.model(val_sources, val_inputs,
                                                                                    val_source_lengths)

                vocabulary_size = val_decoder_outputs.size(-1)
                val_outputs_flat = val_decoder_outputs.view(-1, vocabulary_size)
                val_targets_flat = val_targets.view(-1)
                val_batch_loss = self.loss_function(val_outputs_flat, val_targets_flat)

                val_batch_losses.append(val_batch_loss.item())
                val_batch_token_count = val_targets_flat.size(0)
                val_batch_token_counts.append(val_batch_token_count)

            val_token_counts = sum(val_batch_token_counts)
            val_epoch_loss_per_token = sum(val_batch_losses) / val_token_counts
        else:
            val_epoch_loss_per_token = None

        return train_epoch_loss_per_token, val_epoch_loss_per_token

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            train_epoch_loss_per_token, val_epoch_loss_per_token = self.train(epoch)

            self.train_epoch_losses.append(train_epoch_loss_per_token)
            self.val_epoch_losses.append(val_epoch_loss_per_token)

            train_epoch_perplexity = self._calculate_perplexity(train_epoch_loss_per_token)
            val_epoch_perplexity = self._calculate_perplexity(val_epoch_loss_per_token)

            if epoch % self.print_every == 0 and self.logger:
                self._leave_log(train_epoch_loss_per_token, val_epoch_loss_per_token,
                                train_epoch_perplexity, val_epoch_perplexity, progress=epoch / epochs)

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

        base_filename = '{model_name}_{epoch}_{start_time}.pth'
        checkpoint_filename = base_filename.format(model_name=self.save_name,
                                                   start_time=self.start_time,
                                                   epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)

        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if min(self.val_epoch_losses) == self.val_epoch_losses[-1]:  # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath


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

    return sources_tensor, inputs_tensor, targets_tensor, torch.tensor(source_lengths), torch.tensor(target_lengths)
