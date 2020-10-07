from config import settings
import h5py
import logging
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50


import GloVepreprocessing
import storage
import model


# Get logger
def setupLogging():
    log_file_path = settings.log_config
    with open(log_file_path, 'rt') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)

setupLogging()
logger = logging.getLogger()

# Get embedding matrix
preprocessor = None

try:
    with open(settings.glove_embed_data, 'rb') as handle:
        preprocessor = pickle.load(handle)
except FileNotFoundError:      
    preprocessor = GloVepreprocessing.GloVepreprocessor()
    with open(settings.glove_embed_data, 'wb') as handle:
        print("before pickle dump")
        pickle.dump(preprocessor, handle)

def get_features_model():
    logger = logging.getLogger()
       
    #Load the ResNet50 model
    image_shape=(settings.image_shape[0],settings.image_shape[1],settings.image_shape[2])
    X_input = Input(image_shape, name="image_input")
    resnet_model = resnet50.ResNet50(weights='imagenet',include_top=True, input_tensor= X_input) 
    features_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
    
    # summarize
    print(features_model.summary())
    return features_model

def get_inference_model():
    checkpoint_path = "../data/models/chk/"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    inference_model = model.ShowAndTell(preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.VOCAB_SIZE, 256, 60, preprocessor.weights)
    
    logger.info("loading training  model") 
    #training_model.load_weights(f"../data/models/w_train_{epoch}.saved")
    training_model.load_weights(f"{checkpoint_path}wtrain-{epoch:03d}")
    return inference_model

def predict(image, features_model, inference_model, epoch, image_size=(224,224)):
    
    # Image preparation for features extraction
    img = Image.fromarray(image)
    img = np.asarray(img.resize(size=image_size))
    image_batch = np.expand_dims(img, axis=0)
    image_batch = resnet.preprocess_input(image_batch)

    # get features
    logger.INFO("Extracting features from image")
    features = features_model.predict(image_batch[0], verbose=0)   
    logger.INFO("Description generation") 
    inference_model.load_weights(f"../data/models/w_train_{epoch}.saved")
    caption = preprocessor.generate_description(self, inference_model, features)
    
    return caption
 



    # Test the ResNet50 classification - 1st part of the model
def classification_generation(set_name, index, epoch = 10):

    model = ResNet50(weights='imagenet')
    logging.info('Model loaded. Started serving...')
    
    # Reading image at given index in set_name
    status, image, caption = storage.read_image(set_name, index)
    
    if status != 200:
        return "No image", caption, None
    print(caption)
    
    img = Image.fromarray(image)
    img = img.resize(size=(224, 224))
    
    image_batch = np.expand_dims(img, axis=0)
    image_batch = preprocess_input(image_batch)
      
    preds = model.predict(image_batch)
    pred_class = decode_predictions(preds, top=5)# ImageNet Decode
    logging.info(pred_class)
    result = str(pred_class[0][0][1]) + " " +   str(pred_class[0][1][1]) + " " + str(pred_class[0][2][1])
        
    return result, caption, image_batch[0]


