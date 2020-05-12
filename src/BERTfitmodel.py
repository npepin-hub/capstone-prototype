import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Lambda 
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, LSTM, Embedding, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import model
import BERTpreprocessing


X, Y  = BERTpreprocessing.batch_generator("train", 0, 100)

mymodel = model.TrainShowAndTell(35, 60, 768)
mymodel.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.01), metrics=['accuracy'])

a0 = np.zeros((X["image_input"].shape[0], 60))
c0 = np.zeros((X["image_input"].shape[0], 60))

mymodel.fit([X["image_input"],X["caption_input"].numpy()
             , a0, c0], Y["output"].numpy(), batch_size=10, epochs=9, verbose=2)