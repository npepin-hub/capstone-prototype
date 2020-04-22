import pandas as pd
import numpy as np
import re
import os
import codecs
import logging

import tensorflow as tf
import tensorflow_hub as hub
#from keras import layers
#from keras.models import Model

from bert import bert_tokenization
import storage

BERT_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
MAX_CAPTION_LENGTH = 35

def import_BERT_pretrained_model():
    # Import BERT model specified in the url    
    #module = hub.Module(BERT_URL)
    module = hub.KerasLayer(BERT_URL, trainable=True)
    return module

def BERT_embed(module, sentences, isInputSentence=True):

    # Retrieve vocab file from cache
    vocab_file = module.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = module.resolved_object.do_lower_case.numpy()

    # Create a tokenizer   
    tokenizer = bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    input_ids = tf.keras.layers.Input(shape=(MAX_CAPTION_LENGTH,), dtype=tf.int32,name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(MAX_CAPTION_LENGTH,), dtype=tf.int32,name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(MAX_CAPTION_LENGTH,), dtype=tf.int32,name="segment_ids")

    input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, MAX_CAPTION_LENGTH, isInputSentence)
    
    pooled_output, sequence_output = module([input_ids_vals, input_mask_vals, segment_ids_vals])    

    return pooled_output, sequence_output

def preprocess_sentence(sentence):    
    # Step 1: Add a "." at the end of the sentence. Not all captions seem to have one
    p = re.compile('.*(\.)$')
    if not (p.match(sentence)):
        sentence += "."
        
    # Step 2: Lowercase
    return sentence.lower()

def preprocess_sentences(sentences): 
    sentences = list(map(lambda sentence: preprocess_sentence(sentence), sentences))
    return sentences
    
def convert_sentence_to_features(sentence, tokenizer, max_seq_len, isInputSentence=True):
    tokens = []
    max_seq = max_seq_len-1
    
    # If the sentence is fed into the RNN for training purpose, the [CLS] token is added at the beginning
    if isInputSentence:
        tokens = ['[CLS]']
        max_seq += 1
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq:
        tokens = tokens[:max_seq]
    # If the sentence is used to evaluate the output of the RNN during training, the [SEP] token is added at the end
    if not isInputSentence:    
        tokens.append('[SEP]')
    
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)
    
    return input_ids, input_mask, segment_ids

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20, isInputSentence=True):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    
    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len, isInputSentence)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    
    return all_input_ids, all_input_mask, all_segment_ids

def batch_generator(set_name, start_index, batch_size):
    logger = logging.getLogger()
    X = []
    Y = []
    for idx in range(start_index, start_index + batch_size):
        try:
            status, image, caption = storage.read_image(set_name, idx)  
            if (int(status) == 200):
                X.append(image)
                Y.append(str(caption))
        except KeyError:
            # Ignores files not found - probably an HHTP error when requesting the URL
            logger.info("-- URL# "+str(idxx)+" storing status not found.")
            continue
    
    module = import_BERT_pretrained_model()
    preprocess_captions = preprocess_sentences(Y)
    
    # Embedds captions sent into the LSTM cell - BERT output has two keys `dict_keys(['sequence_output', 'pooled_output'])
    _,in_captions = BERT_embed(module,Y, isInputSentence=True)
    
    # Embedds captions for loss computation - BERT output has two keys `dict_keys(['sequence_output', 'pooled_output'])`
    _,out_captions = BERT_embed(module, Y, isInputSentence=False)
    
    # Return
    X_data = {
        "image_input":np.array(X),
        "caption_input": in_captions
    }
    Y_data = {
        "output": out_captions
    }
    
    #yield(X_data, Y_data )
    return(X_data, Y_data)

