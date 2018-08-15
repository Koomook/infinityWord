from datasets.seq2seq_datasets import Seq2SeqIndexedDataset

from models.transformer import Transformer, TransformerEncoder, TransformerDecoder
from trainers.seq2seq_trainer import Seq2SeqTrainer, seq2seq_collate_fn

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils import get_logger

from dictionaries import BaseDictionary
from trainers.sorted_2d_batch_sampler import Sorted2DBatchSampler, Sorted2DBatchSamplerOnTheFly

dictionary = BaseDictionary.load('base_dictionary')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

embedding = torch.nn.Embedding(num_embeddings=dictionary.vocabulary_size, embedding_dim=512)
encoder = TransformerEncoder(layers_count=3, d_model=512, heads_count=8, d_ff=512, dropout_prob=0.2, embedding=embedding)
decoder = TransformerDecoder(layers_count=3, d_model=512, heads_count=8, d_ff=512, dropout_prob=0.2, embedding=embedding)
model = Transformer(encoder, decoder)

logger = get_logger(log_name='transformer')
train_dataset = Seq2SeqIndexedDataset('train')
val_dataset = Seq2SeqIndexedDataset('val')

train_batch_sampler = Sorted2DBatchSamplerOnTheFly(train_dataset, batch_size=32, drop_last=False, max_length=100)
val_batch_sampler = Sorted2DBatchSamplerOnTheFly(val_dataset, batch_size=100, drop_last=False, max_length=100)

trainer = Seq2SeqTrainer(model=model,
                         train_dataloader=DataLoader(train_dataset,
                                                     batch_sampler=train_batch_sampler,
                                                     collate_fn=seq2seq_collate_fn),
                         val_dataloader=DataLoader(val_dataset,
                                                   batch_sampler=val_batch_sampler,
                                                   collate_fn=seq2seq_collate_fn),
                         loss_function=CrossEntropyLoss(reduce=False),
                         optimizer=Adam(model.parameters()),
                         print_every=1,
                         save_every=10,
                         save_name='transformer_base',
                         device=device,
                         logger=logger)

trainer.run(epochs=10)