import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import yaml
from PIL import Image
from tensorflow.keras.applications import resnet, resnet50
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

sys.path.append('/src/')
#sys.path.append('..')
import extraction
import GloVepreprocessing
import model
from config import settings

def extract_images_from_rawdata(rawdata_file_path, img_size):
    data_df = None

    # Read TSV file
    logger.info("Starting rawdata extraction" + rawdata_file_path)
    data_df = pd.read_table(rawdata_file_path, header = None, names = ['caption', 'url'] )
    
    # Extract images and stores images/captions    
    for index, row in data_df.iterrows(): 
        logger.info("Extracting image " + str(index))
        status, image = extraction.get_image(index, row.url, img_size, rawdata_file_path)        
        if status == 200:
            image_path = os.path.join("/pfs/out", os.path.split(rawdata_file_path)[1]+'-'+str(index)+'.png')
            logger.info("Saving image " + image_path)
            image.save(image_path, "png")


def get_resnet_model():
    #Load the ResNet50 model
    image_shape=(settings.image_shape[0],settings.image_shape[1],settings.image_shape[2])
    X_input = Input(image_shape, name="image_input")
    resnet_model = resnet50.ResNet50(weights='imagenet',include_top=True, input_tensor= X_input) 
    model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
    return model


def extract_features_from_image(model, image_filename, img_size):
# Extracts the features of a given image 
    logger = logging.getLogger()
    
    image = Image.open(image_filename)
    #img = Image.fromarray(image)
    img = np.asarray(image.resize(size=img_size))
    img = np.expand_dims(img, axis=0)
    img = resnet.preprocess_input(img)
            
    # get features
    features = model.predict(img, verbose=0)
    features_path = os.path.join("/pfs/out", os.path.split(image_filename)[1]+".features")
    logger.info("Saving features " + features_path)
 
    with open(features_path, 'wb') as handle:
        pickle.dump(features, handle) 

def train(model_dir):
    learning_rate = 0.01
    loss="categorical_crossentropy"
   
    logger = logging.getLogger() 
    # Get embedding matrix
    preprocessor = GloVepreprocessing.preprocessor_factory()

    # Loads model (empty shell)
    training_model = model.injectAndMerge(preprocessor)
    # Compile model
    training_model.compile(loss=loss, optimizer=Adam(lr = learning_rate))
    logger.info(training_model.summary())

    for dirpath, _, files in os.walk("/pfs/rawdata"):
        for caption_file_path in files:
            batch = preprocessor.pachyderm_dataset(os.path.join(dirpath,caption_file_path),"/pfs/features")
            training_model.fit(batch, epochs=1, verbose=1)
    
    training_model.save(os.path.join("/pfs/out", "model.h5"))

def parse_args():
    parser = argparse.ArgumentParser(description='Train a caption generation model.')
    # command line arguments
    parser.add_argument('--stage', type=str)

    parser.add_argument('--modeldir', type=str)
    return parser.parse_args()
       
       
if __name__ == "__main__":
    args = parse_args()

    # Get logger
    logger = logging.getLogger()
    logger.info("Stage : " + args.stage)

    if args.stage == "extract_images_from_rawdata":
        # walk /pfs/rawdata and extract the image from their url on every file found
        for dirpath, _, files in os.walk("/pfs/rawdata"):
            for file in files:
                extract_images_from_rawdata(os.path.join(dirpath, file), (224,224))
    elif args.stage == "extract_features_from_image":
        model = get_resnet_model()
        # walk /pfs/rawdata and extract the image from their url on every file found
        for dirpath, _, files in os.walk("/pfs/images"):
            for file in files:
                extract_features_from_image(model, os.path.join(dirpath, file),(224,224))
    elif args.stage == "train_model":
        train(args.modeldir)       
    else:
        sys.exit(1)
