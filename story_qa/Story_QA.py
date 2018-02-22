__author__ = 'Weiliang Guo'

'''Trains a memory network on the bAbI dataset.

Please refer to story_qa.md while reading this code.

I modified the example code to follow OOP paradigm and addded more comments.

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs using adam optimizer
rather than rmsprop which was used in the original keras example code.


References:
https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py
https://regexone.com
'''

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import numpy as np
import re


class StoryQAData:
    def __init__(self,f_train, f_test):
        train_stories = self.get_stories(f_train)
        print('Number of training stories:', len(train_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        test_stories = self.get_stories(f_test)
        print('Number of test stories:', len(test_stories))

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        self.vocab_size = len(vocab) + 1
        print('-')
        print('Vocab size:', self.vocab_size, 'unique words')
        self.story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        print('Story max length:', self.story_maxlen, 'words')
        self.query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
        print('Query max length:', self.query_maxlen, 'words')
        print('-')
        print('inputs: integer tensor of shape (samples, max_length)')
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        self.inputs_train, self.queries_train, self.answers_train = self.vectorize_stories(train_stories)
        print('inputs_train shape:', self.inputs_train.shape)
        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', self.queries_train.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', self.answers_train.shape)
        self.inputs_test, self.queries_test, self.answers_test = self.vectorize_stories(test_stories)
        print('inputs_test shape:', self.inputs_test.shape)
        print('queries_test shape:', self.queries_test.shape)
        print('answers_test shape:', self.answers_test.shape)

    @staticmethod
    def tokenize(sent):
        # Return the tokens of a sentence including punctuation.  tokenize('Bob dropped the apple. Where is the apple?')
        # ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']


        # Any subpattern inside a pair of parentheses will be captured as a group. In practice, this can be
        # used to extract information like phone numbers or emails from all sorts of data.

        # \W matches any Non-alphanumeric character

        # +	matches one or more repetitions

        # ? matches either zero or one of the preceding character or group

        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def parse_stories(self, lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.strip()
            # split a line only once by encountering first whitespace
            # eg. "4 John went to the kitchen." will be splitted into "4" and "John went to the kitchen."
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    # map(func, *iterables) makes an iterator that computes the function
                    # using arguments from each of the iterables.
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, only_supporting=False, max_length=None):
        '''Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        f = open(f, 'r')
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
        return data

    def vectorize_stories(self, data):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([self.word_idx[w] for w in story])
            queries.append([self.word_idx[w] for w in query])
            answers.append(self.word_idx[answer])
        return (pad_sequences(inputs, maxlen=self.story_maxlen),
                pad_sequences(queries, maxlen=self.query_maxlen),
                np.array(answers))


class StoryQAModel:
    def __init__(self, f_train='story_qa/qa1_single-supporting-fact_train.txt',
                 f_test='story_qa/qa1_single-supporting-fact_test.txt'):
        self.sd = StoryQAData(f_train, f_test)
        self.story_maxlen = self.sd.story_maxlen
        self.query_maxlen = self.sd.query_maxlen
        self.vocab_size = self.sd.vocab_size
        self.f_train = f_train
        self.f_test = f_test

    def build_model(self):

        # Please read section 2 of  paper "End-To-End Memory Networks" for more details.

        # Some comments of the original keras babi_memnn.py example code are a bit misleading, I changed them to
        # follow the paper as much as possible.

        # placeholders
        # `Input()` is used to instantiate a Keras tensor. Its 1st argument is the tensor's shape
        input_sequence = Input((self.story_maxlen,))
        question = Input((self.query_maxlen,))

        # encoders
        # embed the input set X into input memory vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=self.vocab_size, output_dim=64))
        input_encoder_m.add(Dropout(0.3))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input set X into output memory vectors of size query_maxlen
        output_encoder_c = Sequential()
        output_encoder_c.add(Embedding(input_dim=self.vocab_size, output_dim=self.query_maxlen))
        output_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder_u = Sequential()
        question_encoder_u.add(Embedding(input_dim=self.vocab_size, output_dim=64, input_length=self.query_maxlen))
        question_encoder_u.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)

        # encode memory and questions vectors (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        output_encoded_c = output_encoder_c(input_sequence)
        question_encoded_u = question_encoder_u(question)

        # compute a 'match' between the input memory vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match_um = dot([input_encoded_m, question_encoded_u], axes=(2, 2))
        match_um = Activation('softmax')(match_um)

        # add the match matrix with the output memory vector sequence
        response = add([match_um, output_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded_u])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3)(answer)
        answer = Dense(self.vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print('-')
        print('Compiling...')
        return model

    def train_model(self):
        model = self.build_model()
        # train
        model.fit([self.sd.inputs_train, self.sd.queries_train], self.sd.answers_train, batch_size=32, epochs=120,
                  validation_data=([self.sd.inputs_test, self.sd.queries_test], self.sd.answers_test))


if __name__ == '__main__':
    sq = StoryQAModel()
    sq.train_model()