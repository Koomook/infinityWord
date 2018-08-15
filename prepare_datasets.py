from datasets.base_datasets import ChaptersDataset, ChaptersTokenizedDataset, SentencesDataset, SentencesTokenizedDataset, InputTargetIndexedDataset
from datasets.seq2seq_datasets import OneSeq2SeqDataset, Seq2SeqIndexedDataset
from datetime import datetime
from dictionaries import BaseDictionary
from tokenizers import SoyTokenizer

start_time = datetime.now()

print('ChaptersDataset...', datetime.now()-start_time)
ChaptersDataset.prepare_dataset(num_novels=100)
chapters_dataset = ChaptersDataset(phase='train')
print('chapters_dataset', chapters_dataset[0])

print('SentencesDataset...', datetime.now()-start_time)
SentencesDataset.prepare_dataset()
sentences = SentencesDataset('train')

print('SoyPreprocessor...', datetime.now()-start_time)
SoyTokenizer.prepare_preprocessor(sentences)
tokenizer = SoyTokenizer()

print('ChaptersTokenizedDataset...', datetime.now()-start_time)
ChaptersTokenizedDataset.prepare_dataset(tokenizer)
print(ChaptersTokenizedDataset('train')[0])

print('SentencesTokenizedDataset...', datetime.now()-start_time)
SentencesTokenizedDataset.prepare_dataset()
sentences_tokenized = SentencesTokenizedDataset(phase='train')
print('sentences_dataset', sentences_tokenized[0])

print('BaseDictionary...', datetime.now()-start_time)
dictionary = BaseDictionary()
dictionary.prepare_dictionary(sentences_tokenized)
dictionary.save('base_dictionary')

print('InputTargetIndexedDataset...', datetime.now()-start_time)
InputTargetIndexedDataset.prepare_dataset(dictionary=dictionary)
input_target_indexed_dataset = InputTargetIndexedDataset(phase='val')
print('input_target_indexed_dataset', input_target_indexed_dataset[0])

print('OneSeq2SeqDataset...', datetime.now()-start_time)
OneSeq2SeqDataset.prepare_dataset(source_length=1)
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