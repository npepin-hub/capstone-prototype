from config import settings
from itertools import chain
import numpy as np
import os
import pandas as pd
import pickle
import PIL
from PIL import Image
import re
import urllib
import yaml
import zipfile

import logging
import logging.config

from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import storage
import extraction


class GloVepreprocessor(object):
#########################################################################    
                       # Class Variables #
#########################################################################    
   
    # GloVe dir and url
    glove_dir = settings.glove_dir
    glove_url = settings.glove_url
    
    # Path to the embedding file
    glove_weights_file_path = ""
    
    # Embedding parameters
    MAX_SEQUENCE_LENGTH = settings.MAX_SEQUENCE_LENGTH
    EMBEDDING_SIZE = settings.EMBEDDING_SIZE
    VOCAB_SIZE = 0
    MAX_VOCAB_SIZE= settings.MAX_VOCAB_SIZE
    
    # Additional tokens added to the embeddings to mark the beginning  [CLS] and end [SEP] of captions plus a padding token [PAD] and unknow token [UNK]
    PAD_IDX = 0
    CLS_IDX = 1
    SEP_IDX = 2
    UNKNOWN_IDX = 3
    
    # The embedding matrix (idx of token -> embedding vector), and the corresponding mappings word -> idx and idx -> word
    weights = []
    word2idx = {}
    idx2word =  {}
    
    tokenizer = None
    

    
#########################################################################    
    
    def __init__(self):
        weights = []
        word2idx = {}
        idx2word =  {}
        self.tokenizer = None

        self.setupLogging()        
        self.import_GloVe_files()        
        self.fit_tokenizer()
        self.load_GloVe()

        
    ####################
    # Logging Set up   #
    ####################
    def setupLogging(self):
        with open(settings.log_config, 'rt') as file:
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

    ############################################
    # Fit a Tokenizer on the training captions #
    ############################################
    def fit_tokenizer(self):
        # Read TSV files for captions        
        train_df = pd.read_table(settings.train_raw_data, header = None, names = ['caption', 'url'] )        
        train_df["caption"] = [extraction.clean_caption(caption) for caption in train_df["caption"]]
        
        self.tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE)
        self.tokenizer.fit_on_texts(train_df["caption"])
        
        
    ###############
    # Loads Glove #
    ###############
    def load_GloVe(self):
        wordsintokenizer = self.tokenizer.word_index

        originalembedmatrix = {}
        wordsinoriginalembedmatrix = []    

        with open(self.glove_weights_file_path) as file:       
            for index, line in enumerate(file):
                values = line.split() # Word and weights separated by space
                word = values[0] # Word is first symbol on each line
                wordsinoriginalembedmatrix.append(word)       
                word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
                originalembedmatrix[word] = word_weights
                
        # words in tokenizer that are not in the embedding matrix need to be removed from the tokenizer
        remove = set(wordsintokenizer) - set(wordsinoriginalembedmatrix)              
        # removing the words from the tokenizer so that all the words remaining will exist in the embedding matrix
        commonwords = list(filter(lambda x: x not in remove, wordsintokenizer.keys()))
        
        
        preword2idx = {'[PAD]': self.PAD_IDX, settings.start_seq: self.CLS_IDX, settings.end_seq: self.SEP_IDX, '[UNK]': self.UNKNOWN_IDX}
        preidx2word = { self.PAD_IDX :'[PAD]' , self.CLS_IDX :settings.start_seq , self.SEP_IDX :settings.end_seq , self.UNKNOWN_IDX :'[UNK]'}
        n = len(preword2idx)

        originalword2idx = dict(self.tokenizer.word_index.items())
        originalidx2word = dict(self.tokenizer.index_word.items())
    
        # Assemble the above 4 token to word2idx and idx2word
        self.word2idx = preword2idx
        self.idx2word = preidx2word


        w = []
        # weights - VOCAB_LENGTH x EMBEDDING_DIMENTION matrix
        # Insert 4 additional token: the PAD, CLS, SEP, UNK in embedding matrix
        w.insert(self.PAD_IDX, np.random.randn(self.EMBEDDING_SIZE))
        w.insert(self.CLS_IDX, np.random.randn(self.EMBEDDING_SIZE))
        w.insert(self.SEP_IDX, np.random.randn(self.EMBEDDING_SIZE))
        w.insert(self.UNKNOWN_IDX, np.random.randn(self.EMBEDDING_SIZE))
        
        # Creates the word2idx, idx2word, weights
        for i, word in enumerate(commonwords):
            if i >= (self.MAX_VOCAB_SIZE - n):
                break
            idx = int(i+n)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            w.insert(idx, originalembedmatrix[word])

        # Construct our max vocab 
        self.weights = np.asarray(w, dtype=np.float32)
        self.VOCAB_SIZE = self.weights.shape[0]
               
    #######################
    #   Helper functions  #
    #######################
    # Converts a caption list into a list of list of idx 
    def texts_to_sequences(self, captions):
        sequences = []
        for caption in captions:
            tokens = []
            tokens.extend(text_to_word_sequence(caption))
            if len(tokens) > self.MAX_SEQUENCE_LENGTH - 1:
                tokens = tokens[:self.MAX_SEQUENCE_LENGTH - 1]
                tokens.append(settings.end_seq)
            sequences.append(self.convert_tokens_to_ids(tokens))
        return sequences
    
    # Converts a list of tokens into idx    
    def convert_tokens_to_ids(self, tokens):
        input_ids =[]
        for token in tokens:
            input_ids.append(self.convert_token_to_id(token))
        return input_ids
    
    # Converts a token into idx    
    def convert_token_to_id(self, token):
        try:
            return self.word2idx[token]
        except KeyError:
            return self.word2idx['[UNK]']
        
        
    # Map an integer to a word
    def word_for_id(self, integer):
        word = '[UNK]'
        try:
            word = self.idx2word[integer]
        except KeyError:
            pass    
        return word

   ###############
   #  Generator  #
   ###############
    def generator(self,set_name, batch_size, start_index=0):
        logger = logging.getLogger()        
        batch_start_index = start_index

        while True: 
            X , Yin, Yout = [], [], []
            idx, missed_idx = 0 , 0
            while idx < batch_size:
                try:
                    logger.debug("loading feature " + str(batch_start_index + idx + missed_idx))
                    status, _, features, caption = storage.read_image(set_name, batch_start_index + idx + missed_idx)                        
                    # encode the sequence       
                    seq = self.texts_to_sequences([str(caption)])[0]
                    if (int(status) == 200):

                        # split one sequence into multiple X,y pairs
                        for i in range(1, len(seq)):
                            # split into input and output pair
                            in_seq, out_seq = seq[:i], seq[i]
                            in_seq = pad_sequences([in_seq], maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')[0]
                            # encode output sequence
                            out_seq = to_categorical([out_seq], num_classes=self.VOCAB_SIZE)[0]
                            # store
                            if (in_seq[0] != 1):
                                logger.INFO("IN SEQUENCE: "+ str(in_seq))
                            
                            X.append(features.transpose())
                            Yin.append(in_seq)
                            Yout.append(out_seq)

                        idx = idx + 1
                    else:
                        missed_idx = missed_idx + 1
                except KeyError:
                    # Ignores files not found - probably an HHTP error when requesting the URL
                    missed_idx = missed_idx + 1
                    logger.info("-- URL# "+str(batch_start_index + idx + missed_idx)+" storing status not found.")
                    continue


                # Return prep
                """
                X_data = {
                    "features_input": np.reshape(X, (len(X), 2048)),
                    "caption_input": np.reshape(Yin, (len(Yin), self.MAX_SEQUENCE_LENGTH))

                }
                Y_data = {
                    "caption_output": np.reshape(Yout, (len(Yout), self.VOCAB_SIZE))
                }
                """
            batch_start_index =  batch_start_index + batch_size
            #yield([X_data["features_input"],X_data["caption_input"]],[Y_data["caption_output"]])
            logger.debug(".")
            yield([np.reshape(X, (len(X), 2048)),np.reshape(Yin, (len(Yin), self.MAX_SEQUENCE_LENGTH))],[np.reshape(Yout, (len(Yout), self.VOCAB_SIZE))])


    
    def get_loss_function(self):

        # Masking the ['PAD'] on Loss function -> ie: weights the padded areas to 0 in each caption to focus on "true" words only.
        # TODO: See how to add a Masking Layer as part of the model
        def masked_categorical_crossentropy(y_true, y_pred):
            pad_idx = self.word2idx["[PAD]"]
            y_true_idx = K.argmax(y_true)

            # In each output vector of the LSTM, the idexes at which the value is the padding's index ['PAD'] are set to 0.
            mask = K.cast(K.equal(y_true_idx, pad_idx), K.floatx())
            mask = 1.0 - mask

            loss = K.categorical_crossentropy(y_true, y_pred) * mask

            # The loss is relative to the number of unpadded words in the vector
            return K.sum(loss) / K.sum(mask)

        return masked_categorical_crossentropy
     
        ########################
        #  Caption Generation  #
        ########################                
        # generate a description for the features of an image
    def generate_description(self, model, photo):
        # Seed the generation process
        in_text = settings.start_seq
        # Iterate over the whole length of the sequence
        for i in range(preprocessor.MAX_SEQUENCE_LENGTH):
            # integer encode input sequence
            sequence = preprocessor.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=preprocessor.MAX_SEQUENCE_LENGTH)
            # predict next word
            yhat = model.predict([photo,sequence], verbose=0)
            # convert probability to integer
            yhat = np.argmax(yhat)
            # map integer to word
            word = preprocessor.word_for_id(yhat)

            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word == setting.end_seq:
                break
        return in_text
    
    