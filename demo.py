from datasets.seq2seq_datasets import Seq2SeqIndexedDataset
from models.transformer_encoder import TransformerEncoder
from models.transformer_decoder import TransformerDecoder
import torch
from torch import nn
from utils import get_logger
from dictionaries import BaseDictionary
from models.seq2seq_model import Seq2SeqModel
from generators.base_generators import CandidatesGenerator
from flask import Flask, request
from flask_restplus import Resource, Api, fields

app = Flask(__name__)
api = Api(app)

device = torch.device('cpu')
logger = get_logger(log_name='demo')

dictionary = BaseDictionary.load('base_dictionary')
embeddings = nn.Embedding(num_embeddings=dictionary.vocabulary_size, embedding_dim=512)
encoder = TransformerEncoder(num_layers=3, d_model=512, heads=8, d_ff=512, dropout=0.2, embeddings=embeddings)
decoder = TransformerDecoder(num_layers=3, d_model=512, heads=8, d_ff=512, dropout=0.2, embeddings=embeddings)
model = Seq2SeqModel(encoder, decoder)

generator = CandidatesGenerator(model=model,
                                preprocess=lambda source: [dictionary[word] for word in sum([sentence.split() for sentence in source], [])],
                                postprocess=lambda h: ' '.join([dictionary.idx2word[token_id] for token_id in h[:-1]]), # exclude EndSent token
                                checkpoint_filepath='parameters/Seq2SeqModel/Seq2SeqModel_0_2018-08-05 14:23:30.982203.pth',
                                max_length=30,
                                beam_size=8
                                )

sentences_model = api.model('Sentences', {
    'sentences': fields.List(fields.String, required=True, description='...', example=['버려진 섬마다 꽃이 피었다', '그 꽃은 진달래꽃']),
    'test': fields.String
})


@api.route('/hello')
class Hello(Resource):
    def get(self):
        return {'status': 'success'}


@api.route('/generate_candidates', methods=['PUT'])
class Generator(Resource):
    @api.expect(sentences_model)
    def put(self):
        print(request.json)
        source = request.json['sentences'][-5:]
        generated_candidates = generator.generate_candidates(source, n_candidates=5)
        return {'generated_candidates': generated_candidates}


if __name__ == '__main__':
    app.run(debug=True)