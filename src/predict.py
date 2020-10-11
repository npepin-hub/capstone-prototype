from config import settings
import h5py
import io
import logging
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
import yaml

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.applications import resnet50
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model, Model
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.keras.preprocessing.sequence import pad_sequences

import GloVepreprocessing
import storage
import model

def get_features_model():
    #Load the ResNet50 model
    image_shape=(settings.image_shape[0],settings.image_shape[1],settings.image_shape[2])
    X_input = Input(image_shape, name="image_input")
    resnet_model = resnet50.ResNet50(weights='imagenet',include_top=True, input_tensor= X_input) 
    features_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
    
    return features_model

def get_inference_model(epoch, preprocessor):
    logger = logging.getLogger()

    checkpoint_path = settings.model_path+"chk/"
    inference_model = model.injectAndMerge(preprocessor)

    try:        
        logger.info("Loading model for inference") 
        inference_model.load_weights(f"{checkpoint_path}wtrain-{epoch:03d}")
        #inference_model.load_weights(f"../data/models/w_train_{epoch}.saved")
    except NotFoundError:
        logger.info("No weights to load - Sorry!")
    return inference_model

def predict(image, preprocessor, features_model, inference_model, image_size=(settings.image_shape[0],settings.image_shape[1])):
    logger = logging.getLogger()
    
    # Image preparation for features extraction
    img = np.asarray(image.resize(size=image_size))
    image_batch = np.expand_dims(img, axis=0)
    image_batch = preprocess_input(image_batch)

    # get features
    logger.info("Extracting features from image")
    features = features_model.predict(image_batch, verbose=0)   
    logger.info("Description generation") 
    
    caption = generate_description(inference_model, features, preprocessor)   
    return caption
 

# generate a description for the features of an image
def generate_description(inference_model, photo, preprocessor):
    # Seed the generation process
    in_text = settings.start_seq
    # Iterate over the whole length of the sequence
    for i in range(preprocessor.MAX_SEQUENCE_LENGTH):
        # integer encode input sequence
        sequence = preprocessor.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=preprocessor.MAX_SEQUENCE_LENGTH)
        # predict next word
        prediction = inference_model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(prediction)
        if yhat == preprocessor.UNKNOWN_IDX:  
            prediction = prediction[0].argsort()[-2:][::-1]
            yhat = prediction[1]
        # map integer to word
        word = preprocessor.word_for_id(yhat)
        
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == settings.end_seq:
            break
    # Remove start and end tokens
    caption = in_text.split()[1:-1]
    caption = " ".join(caption)
    return caption

if __name__ == "__main__":
    # Get logger
    logger = logging.getLogger()   
    # Get embedding matrix
    preprocessor = GloVepreprocessing.preprocessor_factory()

    print("Hello")
    image = Image.open("../data/cap.jpg")
  
    #image = io.BytesIO(image)
    epoch = 67    
    features_model = get_features_model()
    inference_model = get_inference_model(epoch, preprocessor)
    caption = predict(image, preprocessor, features_model, inference_model) 
    print(caption)



