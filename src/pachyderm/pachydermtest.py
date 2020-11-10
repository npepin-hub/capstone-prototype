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
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions

sys.path.append('/src/')
#sys.path.append('..')
import extraction
import GloVepreprocessing
import model
import predict as inference
from config import settings


'''
    Image extraction from urls
'''
def extract_images_from_rawdata(rawdata_file_path, img_size):
    data_df = None

    # Read TSV file
    logger.info("Starting rawdata extraction" + rawdata_file_path)
    data_df = pd.read_table(rawdata_file_path, header = None, names = ['caption', 'url'] )
 
    image_dir_path = os.path.join("/pfs/out", os.path.split(rawdata_file_path)[1])
    os.makedirs(image_dir_path, exist_ok=True) 
    # Extract images from url  
    for index, row in data_df.iterrows(): 
        logger.info("Extracting image " + str(index))
        status, image = extraction.get_image(index, row.url, img_size, rawdata_file_path)        
        if status == 200:
            image_path = os.path.join(image_dir_path,str(index)+'.png')
            logger.info("Saving image " + image_path)
            image.save(image_path, "png")


def get_features_model():
    #Load the pretrained ResNet50 model for features extraction
    image_shape=(settings.image_shape[0],settings.image_shape[1],settings.image_shape[2])
    X_input = Input(image_shape, name="image_input")
    resnet_model = resnet50.ResNet50(weights='imagenet',include_top=True, input_tensor= X_input) 
    features_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
    
    return features_model

def get_inference_model(model_path, preprocessor):
    #Load the trained model for inference
    logger = logging.getLogger()
    logger.info("Getting the inference model: " + model_path)
    
    try:        
        logger.info("Loading model for inference") 
        #inference_model.load_weights(f"{checkpoint_path}wtrain-{epoch:03d}")
        inference_model = load_model(model_path)
        return inference_model
    except Exception as e:
        logger.info("No weights to load - Sorry!")
        logger.info(e)
        inference_model = model.injectAndMerge(preprocessor)
        return inference_model
    
    


'''
    Features extraction from Images
'''
def extract_features_from_image(model, image_filename, img_size):
# Extracts the features of a given image 
    logger = logging.getLogger()
    
    image = Image.open(image_filename)
    #img = Image.fromarray(image)
    img = np.asarray(image.resize(size=img_size))
    img = np.expand_dims(img, axis=0)
    img = resnet.preprocess_input(img)

    image_name = os.path.split(image_filename)[1] # 1.png
    image_tail = os.path.split(image_filename)[0] # /pfs/images/000000000
    features_dir_path = os.path.join("/pfs/out", os.path.split(image_tail)[1]) #/pfs/out/00000000 
    os.makedirs(features_dir_path, exist_ok=True) 
            
    # get features
    features = model.predict(img, verbose=0)
    features_path = os.path.join(features_dir_path, image_name + ".features")
    logger.info("Saving features " + features_path)
 
    with open(features_path, 'wb') as handle:
        pickle.dump(features, handle) 


'''
    Gathering of data to feed the model
'''
def consolidate():
    for caption_dir_name, _, files in os.walk("/pfs/rawdata"):
        for caption_file_name in files:
            bucket = caption_file_name
            # create repository named after each caption file /pfs/out/00000000 
            bucket_dir_path = os.path.join("/pfs/out", bucket) 
            os.makedirs(bucket_dir_path, exist_ok=True)
            # Soft link for caption files 
            src = os.path.join(caption_dir_name, caption_file_name) 
            dst = os.path.join(bucket_dir_path, "captions.tsv") 
            os.symlink(src, dst)
                          
            data_df = pd.read_table(src, header = None, names = ['caption', 'url'] )   
            for index, _ in data_df.iterrows(): 
                # Soft link for features files 
                features_src = os.path.join("/pfs/features", bucket, str(index)+'.png.features')
                features_dst = os.path.join(bucket_dir_path, str(index)+'.png.features')
                try:
                    with open(features_src, 'rb') as handle:
                        os.symlink(features_src, features_dst)
                except Exception as e: 
                    logger.info(e)     
                    continue
                                    
    
'''
    Training of the model
'''
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

    # Create a callback that saves the model's weights
    checkpoint_filepath = '/pfs/out/checkpoints'
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss', 
    save_freq=500,
    mode='auto',
    verbose=1,
    save_best_only=False)

    generator = preprocessor.pachyderm_dataset("/pfs/consolidate", batch_size=10)
    training_model.fit(generator, epochs=1, verbose=1, callbacks=[model_checkpoint_callback], workers=1)
    # If all goes well, saves the complete model in a new repo - If not, we will need to retrieve the last checkpoint
    os.makedirs("/pfs/out/saved", exist_ok=True) 
    #exported_model_path = os.path.join(model_dir, "saved_model.h5") 
    #logger.info("Saving the trained model in " + exported_model_path)
    training_model.save("/pfs/out/saved/saved_model.h5")

'''
    Caption prediction of an image
'''
def predict(image_path, preprocessor, features_model, inference_model, image_size=(settings.image_shape[0],settings.image_shape[1])):
    logger = logging.getLogger()
    logger.info("Entering the predict function.")
    
    # Image preparation for features extraction
    image = Image.open(image_path)
    img = np.asarray(image.resize(size=image_size))
    image_batch = np.expand_dims(img, axis=0)
    image_batch = preprocess_input(image_batch)

    # get features
    logger.info("Extracting features from image")
    features = features_model.predict(image_batch, verbose=0)   
    logger.info("Description generation") 
    
    # retrieve image name
    image_name = os.path.split(image_path)[1]
    caption = inference.generate_description(inference_model, features, preprocessor)
    prediction_path = os.path.join("/pfs/out", image_name+".txt")
    logger.info("Saving the caption prediction " + prediction_path)
 
    with open(prediction_path, "w") as file:
        file.write(caption)
   
    return caption


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
        # walks /pfs/rawdata and extract the image from their url on every file found
        for dirpath, _, files in os.walk("/pfs/rawdata"):
            for file in files:
                extract_images_from_rawdata(os.path.join(dirpath, file), (224,224))
    elif args.stage == "extract_features_from_image":
        model = get_features_model()
        # walks /pfs/images and extract the features for each image found
        for dirpath, _, files in os.walk("/pfs/images"):
            for file in files:
                extract_features_from_image(model, os.path.join(dirpath, file),(224,224))
    elif args.stage == "consolidate":
        # aggregates data from /pfs/rawdata(captions) and /pfs/features(features) to feed the model
        consolidate()       
    elif args.stage == "train_model":
        # A first pass at training //todo checkpoints callback and load of a given model to retrain 
        train("/pfs/savedmodel")
    elif args.stage == "predict":
        logger = logging.getLogger() 
        # Get embedding matrix
        preprocessor = GloVepreprocessing.preprocessor_factory()
        # Get models
        features_model = get_features_model()
        #inference_model = get_inference_model("/pfs/model/saved/saved_model.h5", preprocessor)
        inference_model = get_inference_model("/pfs/savedmodel/saved_model.h5", preprocessor)
        
        # Predict the caption from the given image using the trained model //todo choose what version of the model should be loaded
        for dirpath, _, files in os.walk("/pfs/inpredict"):
            for file in files:
                predict(os.path.join(dirpath, file), preprocessor, features_model, inference_model)      
    else:
        sys.exit(1)
