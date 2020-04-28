import tensorflow as tf
import pandas as pd
import zipfile
import os
import numpy as np
import urllib
import re
import yaml
import logging
import logging.config

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
from keras.initializers import Constant
import keras.backend as K

import storage


class GloVepreprocessor(object):
#########################################################################    
                       # Class Variables #
#########################################################################    
   
    # GloVe dir and url
    glove_dir = "../data/glove"
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    # Path to the embedding file
    glove_weights_file_path = ""
    
    # Embedding parameters
    MAX_SEQUENCE_LENGTH = 35
    EMBEDDING_SIZE = 50
    VOCAB_SIZE = 0
    
    # Additional tokens added to the embeddings to mark the beginning  [CLS] and end [SEP] of captions plus a padding token [PAD] and unknow token [UNK]
    PAD_IDX = 0
    CLS_IDX = 1
    SEP_IDX = 2
    UNKNOWN_IDX = 3
    
    # The embedding matrix (idx of token -> embedding vector), and the corresponding mappings word -> idx and idx -> word
    weights = []
    word2idx = {}
    idx2word =  {}
    
    Tokenizer = None
    

    
#########################################################################    
    
    def __init__(self):
        self.setupLogging()
        self.import_GloVe_files()
        self.load_GloVe()
        self.fit_tokenizer()

        
    ####################
    # Logging Set up   #
    ####################
    def setupLogging(self):
        with open('../config/logConfig.yml', 'rt') as file:
            config = yaml.safe_load(file.read())
            logging.config.dictConfig(config)
            

    ########################
    # Retrieve Glove Files #
    ########################
    def import_GloVe_files(self):
        # Create glove directory
        if not os.path.isdir(self.glove_dir):
            os.makedirs(self.glove_dir)

        self.glove_weights_file_path = os.path.join(self.glove_dir, f'glove.6B.{self.EMBEDDING_SIZE}d.txt')

        if not os.path.isfile(self.glove_weights_file_path):

            # Glove embedding weights can be downloaded from https://nlp.stanford.edu/projects/glove/    
            local_zip_file_path = os.path.join(self.glove_dir, os.path.basename(self.glove_url))

            if not os.path.isfile(local_zip_file_path):
                print(f'Retrieving glove weights from {self.glove_url}')
                urllib.request.urlretrieve(self.glove_url, local_zip_file_path)

            with zipfile.ZipFile(local_zip_file_path, 'r') as z:
                print(f'Extracting glove weights from {local_zip_file_path}')
                z.extractall(path=self.glove_dir)


    ###############
    # Loads Glove #
    ###############
    def load_GloVe(self):
        # converts a sequence of words to sequence of integers for embedding lookup

        self.word2idx = { '[PAD]': self.PAD_IDX, '[CLS]': self.CLS_IDX, '[SEP]': self.SEP_IDX, '[UNK]': self.UNKNOWN_IDX}
        self.idx2word = { self.PAD_IDX :'[PAD]' , self.CLS_IDX :'[CLS]' , self.SEP_IDX :'[SEP]' , self.UNKNOWN_IDX :'[UNK]'}

        # Insert 4 additional token: the PAD, CLS, SEP, UNK in embedding matrix
        self.weights.insert(self.PAD_IDX, np.random.randn(self.EMBEDDING_SIZE))
        self.weights.insert(self.CLS_IDX, np.random.randn(self.EMBEDDING_SIZE))
        self.weights.insert(self.SEP_IDX, np.random.randn(self.EMBEDDING_SIZE))
        self.weights.insert(self.UNKNOWN_IDX, np.random.randn(self.EMBEDDING_SIZE))

        # idx2word - decodes an integer sequence to words  
        # weights - VOCAB_LENGTH x EMBEDDING_DIMENTION matrix

        with open(self.glove_weights_file_path) as file:
            for index, line in enumerate(file):
                values = line.split() # Word and weights separated by space
                word = values[0] # Word is first symbol on each line
                word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
                self.word2idx[word] = index + 4 # PAD, CLS, SEP, UNK are predefined tokens ->  shift index by 4
                self.idx2word[index + 4 ] = word
                self.weights.append(word_weights)

        # Construct our final vocab
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.VOCAB_SIZE=self.weights.shape[0]

    ############################################
    # Fit a Tokenizer on the training captions #
    ###############
    def fit_tokenizer(self):
        # Read TSV files for captions
        train_df = pd.read_table('../data/Train_GCC-training.tsv', header = None, names = ['caption', 'url'] )

        train_df["caption"] = [self.preprocess_sentence(caption) for caption in train_df["caption"]]
        
        if not self.VOCAB_SIZE == 0:
            self.tokenizer = Tokenizer(num_words=self.VOCAB_SIZE)
            self.tokenizer.fit_on_texts(train_df["caption"])
        else:
            self.load_GloVe()
            self.fit.tokenizer()
       
       
    # Apply a first set of filter to a caption
    def preprocess_sentence(self, sentence):    
        # Step 1: Add a "." at the end of the sentence if no punctuation. 
        p = re.compile('.*([\.?!])$')
        if not (p.match(sentence)):
            sentence += "."

        # Step 2: Lowercase
        return sentence.lower()
    
    def preprocess_sentences(self, sentences): 
        sentences = list(map(lambda sentence: self.preprocess_sentence(sentence), sentences))
        return sentences
    
    # Converts a list of token into idx    
    def convert_tokens_to_ids(self, tokens):
        input_ids =[]
        for token in tokens:
            try:
                input_ids.append(self.word2idx[token])
            except KeyError:
                input_ids.append(self.word2idx['[UNK]'])

        return input_ids
    
    #  Tokenize a given sentence, return a list of the tokens ids
    def get_sentence_ids(self, sentence, tokenizer, isInputSentence=True):
        tokens = []
        max_seq = self.MAX_SEQUENCE_LENGTH-1

        # If the sentence is fed into the RNN for training purpose, the [CLS] token is added at the beginning
        if isInputSentence:
            tokens = ['[CLS]']
            max_seq += 1
        tokens.extend(text_to_word_sequence(sentence))
        if len(tokens) > max_seq:
            tokens = tokens[:max_seq]
        # If the sentence is used to evaluate the output of the RNN during training, the [SEP] token is added at the end
        if not isInputSentence:    
            tokens.append('[SEP]')

        input_ids = self.convert_tokens_to_ids(tokens)    
        input_ids = pad_sequences([input_ids], padding='post', truncating='post', maxlen=self.MAX_SEQUENCE_LENGTH)

        return input_ids

    def get_sentences_ids(self, sentences, tokenizer, isInputSentence=True):
        sentences_input_ids = []

        for sentence in sentences:
            input_ids = self.get_sentence_ids(sentence, tokenizer, isInputSentence)
            sentences_input_ids.append(input_ids)

        return sentences_input_ids
    
    # Given a list of captions, returns an np.array of their corresponding embeddings. 
    # return shape is (batch_size, MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE)
    def GloVe_embed(self, list_of_captions, weights, isInputSentence=True):
        embeddings = []
    
        list_of_captions_ids = self.get_sentences_ids(self.preprocess_sentences(list_of_captions), self.tokenizer, isInputSentence)
    
        for caption_ids in list_of_captions_ids:
            for id in caption_ids:
                caption_embedding = []
                caption_embedding.append(self.weights[id])
            embeddings.extend(caption_embedding)
    
        return embeddings
    
    # Given a set_name ("train" or "validation"), retrieves a batch-size of image/captions
    # Returns the batch-size of images and embedded captions (2 sets: embedded captions fed as an imput of the RNN and those used to estimate the loss)
    def batch_generator(self, set_name, start_index, batch_size):
        logger = logging.getLogger()
        X = []
        Y = []
        idx = 0
        stop_index = start_index + batch_size
        while idx < stop_index:
            try:
                status, image, caption = storage.read_image(set_name, idx)  
                if (int(status) == 200):
                    X.append(image)
                    Y.append(str(caption))
                else:
                    stop_index += 1
                idx = idx + 1
            except KeyError:
                # Ignores files not found - probably an HHTP error when requesting the URL
                logger.info("-- URL# "+str(idx)+" storing status not found.")
                continue


        # Embedds captions sent into the LSTM cell - BERT output has two keys `dict_keys(['sequence_output', 'pooled_output'])
        in_captions = self.GloVe_embed(Y, self.weights, isInputSentence=True)

        # Embedds captions for loss computation - BERT output has two keys `dict_keys(['sequence_output', 'pooled_output'])`
        out_captions = self.GloVe_embed(Y, self.weights, isInputSentence=False)

        # Return
        X_data = {
            "image_input":np.array(X),
            "caption_input": np.reshape(in_captions, (batch_size, self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_SIZE))
        }
        Y_data = {
            "caption_output": np.reshape(out_captions, (batch_size, self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_SIZE))
        }

        #yield(X_data, Y_data )
        return(X_data, Y_data)
    
    
    