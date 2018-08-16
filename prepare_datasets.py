from datasets.base_datasets import ChaptersDataset, ChaptersTokenizedDataset, SentencesDataset, SentencesTokenizedDataset, InputTargetIndexedDataset
from datasets.seq2seq_datasets import OneSeq2SeqDataset, Seq2SeqIndexedDataset
from datetime import datetime
from dictionaries import BaseDictionary
from tokenizers import SoyTokenizer

start_time = datetime.now()

print('ChaptersDataset...', datetime.now()-start_time)
ChaptersDataset.prepare_dataset(num_novels=1000)
chapters_dataset = ChaptersDataset(phase='train')
print('chapters_dataset', chapters_dataset[0])

print('SentencesDataset...', datetime.now()-start_time)
SentencesDataset.prepare_dataset()
sentences = SentencesDataset('train')

print('SoyPreprocessor...', datetime.now()-start_time)
SoyTokenizer.prepare_preprocessor(sentences, min_count=3,
                               min_cohesion_forward=0.05,
                               min_right_branching_entropy=0.0)
tokenizer = SoyTokenizer()

print('ChaptersTokenizedDataset...', datetime.now()-start_time)
ChaptersTokenizedDataset.prepare_dataset(tokenizer)
print(ChaptersTokenizedDataset('train')[0])

print('SentencesTokenizedDataset...', datetime.now()-start_time)
SentencesTokenizedDataset.prepare_dataset()
sentences_tokenized = SentencesTokenizedDataset(phase='train')
print('sentences_dataset', sentences[0], sentences_tokenized[0])
print('sentences_dataset', sentences[10], sentences_tokenized[10])
print('sentences_dataset', sentences[4500], sentences_tokenized[4500])
print('sentences_dataset', sentences[10000], sentences_tokenized[10000])

print('BaseDictionary...', datetime.now()-start_time)
dictionary = BaseDictionary()
dictionary.prepare_dictionary(sentences_tokenized, min_count=20)
dictionary.save('base_dictionary')
print('vocabulary_size', dictionary.vocabulary_size)

print('InputTargetIndexedDataset...', datetime.now()-start_time)
InputTargetIndexedDataset.prepare_dataset(dictionary=dictionary)
input_target_indexed_dataset = InputTargetIndexedDataset(phase='val')
print('input_target_indexed_dataset', input_target_indexed_dataset[0])

print('OneSeq2SeqDataset...', datetime.now()-start_time)
OneSeq2SeqDataset.prepare_dataset(source_length=3)
one_seq2seq_dataset = OneSeq2SeqDataset('train')
print('one_seq2seq_dataset[0]', one_seq2seq_dataset[0])

print('Seq2SeqIndexedDataset...', datetime.now()-start_time)
Seq2SeqIndexedDataset.prepare_dataset(dictionary)
seq2seq_indexed_dataset = Seq2SeqIndexedDataset('train')
print('seq2seq_indexed_dataset[0]', seq2seq_indexed_dataset[0])

end_time = datetime.now()
print('start_time', start_time)
print('end_time', end_time)
print('took', end_time - start_time)