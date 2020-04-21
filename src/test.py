import numpy as np
import logging

import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Lambda 
from keras.layers import Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, LSTM, Embedding, TimeDistributed
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import keras.backend as K

import model
import preprocessing


X, Y  = preprocessing.batch_generator("train", 0, 100)

mymodel = model.TrainShowAndTell(35, 60, 768)
mymodel.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.01), metrics=['accuracy'])

a0 = np.zeros((X["image_input"].shape[0], 60))
c0 = np.zeros((X["image_input"].shape[0], 60))

mymodel.fit([X["image_input"],X["caption_input"]
             , a0, c0], Y["output"], batch_size=10, epochs=9, verbose=2)