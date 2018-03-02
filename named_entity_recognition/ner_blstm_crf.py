__author__ = 'Weiliang Guo'

'''
Use pre-trained word embeddings for trainning, validating, testing, saving the blstm-crf model and tagging new sentence with ner tags.
Reference:
https://github.com/Hironsan/anago/blob/master/examples/ner_glove.py
'''

import os

import anago
from anago.reader import load_data_and_labels, load_glove


EMBEDDING_PATH = 'Replace this with your downloaded pre-trained embeddings path'

train_path = 'named_entity_recognition/train.txt'
valid_path = 'named_entity_recognition/valid.txt'

print('Loading data...')
x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)
print(len(x_train), 'train sequences')
print(len(x_valid), 'valid sequences')

embeddings = load_glove(EMBEDDING_PATH)

# As usual, you may need to tune the hyper-parameters w.r.t. the blstm used in this model.
# Run print(help(anago.Sequence)) to see more details.

# Use pre-trained word embeddings, default of word_emb_size is 100, change it to be
# the same as the pretrained embeddings, Here I use a 300 version.

# model = anago.Sequence(embeddings=embeddings, word_emb_size=300)
# model.train(x_train, y_train, x_valid, y_valid)

# Your trained files of the model including config.json, model_weights.h5, preprocessor.pkl will be
# saved in the specified path
# model.save(dir_path='specify where to save your trained model')

model = anago.Sequence.load(dir_path='specify where to load your trained model')

words = '"We started running, many of the girls were screaming," Fatima said. "We were running towards the gate.'.split()

print(model.analyze(words=words))