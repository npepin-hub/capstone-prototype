from config import settings
import logging
import numpy as np


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, add, Dense, Activation, RepeatVector, Dropout
from tensorflow.keras.layers import Reshape, LSTM, Embedding, TimeDistributed, GlobalAveragePooling2D

from tensorflow.keras.models import Model



def injectAndMerge(preprocessor):

    #############################
    # Image features extraction #
    #############################
    X_input = Input(shape=(2048,), name="features_input")
    #X = GlobalAveragePooling2D()(X)
    
    X = Dropout(0.5)(X_input)
    X = Dense(preprocessor.EMBEDDING_SIZE, activation='relu')(X)
    X = RepeatVector(preprocessor.MAX_SEQUENCE_LENGTH)(X)    
    

    ####################################
    # Language Model Captions encoding #
    ####################################
    caption = Input(shape=(preprocessor.MAX_SEQUENCE_LENGTH,), name="caption_input")  
    #Y = Embedding(input_dim=preprocessor.VOCAB_SIZE, output_dim=preprocessor.EMBEDDING_SIZE, name="wordembed")(caption)
    Y = Embedding(input_dim=preprocessor.VOCAB_SIZE, output_dim=preprocessor.EMBEDDING_SIZE, mask_zero=True, name="wordembed")(caption)
    Y = Dropout(0.5)(Y)
    Y = LSTM(256, return_sequences=True)(Y)
    Y = TimeDistributed(Dense(preprocessor.EMBEDDING_SIZE))(Y)
 
    
    ###################
    # Agregated Model #
    ###################
    XY = add([X, Y])
    XY = LSTM(128, return_sequences=True)(XY)
    XY = LSTM(512, return_sequences=False)(XY)
    out = Dense(preprocessor.VOCAB_SIZE, activation='softmax', name="caption_output")(XY)

    model = Model(inputs=[X_input, caption], outputs = out)
    
    # Set embedding matrix
    model.get_layer("wordembed").set_weights([preprocessor.weights])
    model.get_layer("wordembed").trainable = False
    
    # compile model
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

