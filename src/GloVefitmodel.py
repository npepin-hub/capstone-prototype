
import logging
import numpy as np

import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Lambda 
from keras.layers import Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, LSTM, Embedding, TimeDistributed
from keras.models import Model, load_model
from keras.preprocessing import image 
from keras.optimizers import Adam
from keras.initializers import Constant, glorot_uniform
import keras.backend as K

import GloVepreprocessing
import model

preprocessor = GloVepreprocessing.GloVepreprocessor()

X, Y  = preprocessor.batch_generator("train", 0, 100)


trainmodel,_,_ = model.ShowAndTell(preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.VOCAB_SIZE, preprocessor.EMBEDDING_SIZE, 60, preprocessor.weights)

trainmodel.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.01), metrics=['accuracy'])

a0 = np.zeros((X["image_input"].shape[0], 60))
c0 = np.zeros((X["image_input"].shape[0], 60))

trainmodel.fit([X["image_input"],X["caption_input"]
             , a0, c0], Y["caption_output"], batch_size=10, epochs=10, verbose=2)