import os
from tokenizers import SoyTokenizer
import argparse
from generators.base_generators import CandidatesTransformerGenerator
from models.transformer import Transformer, TransformerEncoder, TransformerDecoder
from dictionaries import BaseDictionary
import torch

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--tokenizer', default='parameters/soypreprocessor/scores.pkl')
parser.add_argument('--dictionary', default='parameters/base_dictionary/base_dictionary')
parser.add_argument('--model', default='parameters/transformer_base/checkpoint.pth')
args = parser.parse_args()

LAYERS_COUNT = 3
D_MODEL = 512
HEADS_COUNT = 8
D_FF = 1024
DROPOUT_PROB = 0.2
MAX_LENGTH = 30
BEAM_SIZE = 8
NUM_CANDIDATES = 5

tokenizer = SoyTokenizer(args.tokenizer)
dictionary = BaseDictionary.load(parameters_filepath=args.dictionary)

embedding = torch.nn.Embedding(num_embeddings=dictionary.vocabulary_size, embedding_dim=D_MODEL)

encoder = TransformerEncoder(layers_count=LAYERS_COUNT, d_model=D_MODEL, heads_count=HEADS_COUNT, d_ff=D_FF, dropout_prob=DROPOUT_PROB, embedding=embedding)
decoder = TransformerDecoder(layers_count=LAYERS_COUNT, d_model=D_MODEL, heads_count=HEADS_COUNT, d_ff=D_FF, dropout_prob=DROPOUT_PROB, embedding=embedding)
model = Transformer(encoder, decoder)
model.eval()
generator = CandidatesTransformerGenerator(model=model,
                                           preprocess=dictionary.index_chapter,
                                           postprocess=lambda h: ' '.join([dictionary.idx2word[token_id] for token_id in h]),
                                           checkpoint_filepath=args.model,
                                           max_length=MAX_LENGTH,
                                           beam_size=BEAM_SIZE
                                           )

source = ['버려진 섬마다 꽃이 피었다', '꽃피는 숲에 저녁 노을이 비치어']
source_tokenized = tokenizer.tokenize_chapter(source)
candidates = generator.generate_candidates(source_tokenized, num_candidates=NUM_CANDIDATES, is_start=True)

print(candidates)