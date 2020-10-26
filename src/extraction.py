from config import settings
import h5py
import io
from itertools import islice
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import PIL
from PIL import Image
import requests
import string
from threading import BoundedSemaphore
import time

from tensorflow.keras.applications import resnet50, resnet
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import concurrent.futures
import storage

#####################################################################################################################
#   This file provides the functions that extract the images and captions from the raw data (tsv files containing   #
#   urls and their caption). It relies on functions provided by the storage module to store those extracted info in #
#   .h5 files. The data in those .h5 files will then be fed to a model for training and validation purposes.        #
#####################################################################################################################

def get_image(index, url, size, set_name):
    """ 
    Retrieves the (padded to the given size) image for a given url.
    Parameters:
    ---------------
    index       the index of the url/caption in the raw data file
    url         the url to the image
    size        image's target dims in the form of (height, width) tuple
    set_name    validate or train

    Returns:
    ----------
    status      http status code
    image       image array
    """
    logger = logging.getLogger()
    logger.info(set_name+"-- Fetching URL#"+str(index))

    # Gets URLs
    try:
        r = requests.get(url, timeout=(10,30))

        logger.info(set_name+"-- URL#"+str(index)+" Http code: "+str(r.status_code))
        if (r.status_code == 200):
            logger.info(url)
            img = Image.open(io.BytesIO(r.content))
            img.thumbnail(size, Image.ANTIALIAS)

            padded_image = Image.new("RGB", size)
            padded_image.paste(img, (int((size[0] - img.size[0])/2), int((size[1] - img.size[1])/2)))

            return int(r.status_code), padded_image
        else:
            return int(r.status_code), None

    except (requests.exceptions.ConnectionError, requests.exceptions.InvalidURL, requests.exceptions.SSLError, requests.exceptions.ContentDecodingError) as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))

    except PIL.UnidentifiedImageError as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))

    except OSError as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))
        
    except Exception as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))


    return 500, None 

    
def store_handler(set_name, index, caption, status_code=200, padded_image=None, features=None):
    """ 
    Stores a caption, http status code, and padded image using the storage module.
    Parameters:
    ---------------
    set_name      validate or train
    index         the index of the url/caption in the raw data file
    caption       the caption of the image
    status-code   the http request status code
    padded_image  the image

    """
        
    logger = logging.getLogger()
    logger.info(set_name+"-- Store Handler Index#"+str(index)+" Http code: "+str(status_code))

    if (int(status_code) == 200):
        storage.store_image(set_name, index, padded_image, features, caption)            
    else:
        storage.store_status(set_name, index, str(status_code)) 
    return 



def request_data_and_store(dataframe, size, set_name, start_index = 0, extraction_size=100000):
    """ 
    Given a panda dataframe containing a list of urls and their corresponding caption, retrieves 
    the images and stores each thumbnailed-padded-to-size Image/Caption using the storage module.
    Note that this function uses multithreading to boost the acquisition of the images.
    Parameters:
    ---------------
    dataframe    the pandas raw dataset
    size         the targeted size of the image in the form of a tuple (height, width)
    set_name     validate or train
    start_index  the index at which the insertion will start in the .h5 files

    Returns:     
    ----------
    nothing for now...
    """
    logger = logging.getLogger()
   
    queue = BoundedSemaphore(100)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:          
        for index, row in islice(dataframe.iterrows(), start_index, extraction_size + start_index):                       
            if (index % 100) == 0:
                logger.info("extraction:request_data_and_store--Processing " + str(index))

            if not(storage.exist(set_name ,index)):
                response_handler =  \
                    lambda future, index=index, caption=row.caption: store_handler(set_name, index, caption, future.result()[0],   future.result()[1], None)

                release_handler = lambda future: queue.release()
                
                logger.debug("extraction:request_data_and_store-- Before ACQUIRE -- " + str(index))
                queue.acquire()
                
                logger.debug("extraction:request_data_and_store-- Before SUBMIT -- " + str(index))
                future_image = executor.submit(get_image, index, row.url, size, set_name)
                future_image.add_done_callback(response_handler)
                future_image.add_done_callback(release_handler)

    return


def extract_resnet_features_and_store(set_name, batch_start_index, batch_end_index):
    """ 
    Given a start and end index, retrieves the features of the images previously stored in a "set_name".h5 files.
    Note that this function uses multithreading to boost both the acquisition of the images and writing the features in a             separate .h5 file.
    Parameters:
    ---------------
    set_name           validate or train
    batch_start_index  the index at which the insertion will start in the .h5 files
    batch_end_index    the index at which the insertion should stop
    
    Returns:     
    ----------
    nothing for now...
    """
    logger = logging.getLogger()
       
    #Load the ResNet50 model
    image_shape=(settings.image_shape[0],settings.image_shape[1],settings.image_shape[2])
    X_input = Input(image_shape, name="image_input")
    resnet_model = resnet50.ResNet50(weights='imagenet',include_top=True, input_tensor= X_input) 
    model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
    
    # summarize
    logger.debug(model.summary())
    
    features = dict()
    index = batch_start_index
    queue = BoundedSemaphore(100)
    features_file_set_name = "features/resnet_train"
    
    if "train" in set_name:
        features_file_set_name = "features/resnet_train"         
    elif "validate" in set_name:
        features_file_set_name = "features/resnet_validate" 
        
    logger.info("extraction:extract_resnet_features_and_store-- Features will be written in -- " + features_file_set_name)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:          
        while (index < batch_end_index):                      

            if (index % 1000) == 0:
                logger.info("extraction:extract_resnet_features_and_store--Before features extraction: " + str(index))
            
            if not(storage.exist(features_file_set_name ,index)):
                response_handler =  \
                                    lambda future, index=index: store_handler(features_file_set_name, index, future.result()[0], future.result()[1], None, future.result()[2])

                release_handler = lambda future: queue.release()

                logger.info("extraction:extract_resnet_features_and_store-- Before queue ACQUIRE -- " + str(index))
                queue.acquire()
                logger.info("extraction:extract_resnet_features_and_store-- Before executor SUBMIT -- " + str(index))
                future_feature = executor.submit(get_image_resnet_features, index, model, set_name)
                future_feature.add_done_callback(response_handler)
                future_feature.add_done_callback(release_handler)

            index += 1

    return


def get_image_resnet_features(index, model, set_name, image_size=(settings.image_shape[0], settings.image_shape[1])):
    """ 
    Extract the features of the image at the given index of the storage .h5 file using a resnet model
    Parameters:
    ---------------
    index       the index of the url/caption in the raw data file
    model       the pretrained resnet 
    image_size  image's target dims in the form of (height, width) tuple
    set_name    validate or train


    Returns:
    ----------
    features    the extracted features
    index       the index of the url/caption in the raw data file
    set_name    validate or train

    """
    logger = logging.getLogger()
    logger.info("extraction:get_image_resnet_features: "+set_name+ " index:" +str(index)+" extracting ")


    try:
        status, image, features, caption = storage.read_image(set_name, index)                   

        if (int(status) == 200):
            caption = final_caption(str(caption))
            
            img = Image.fromarray(image)
            img = np.asarray(img.resize(size=image_size))
            img = np.expand_dims(img, axis=0)
            img = resnet.preprocess_input(img)
            
            # get features
            features = model.predict(img, verbose=0)   
            logger.info("extraction:get_image_resnet_features: "+set_name+ " index:" +str(index)+" features extracted ")
            return caption, status, features
    
    except KeyError:
        # Ignores files not found - probably an HHTP error when requesting the URL
        logger.info("extraction:get_image_resnet_features: "+set_name+ " index:" +str(index)+" Image not present in file. ")

    return caption, status, None 


##################################################################################################################
#  Captions preprocessing                                                                                        #
#  Functions cleaning the initial captions and adding starting and end tokens prior to the training of the model #
#  Might be moved to a dedicated module later...                                                                 #
##################################################################################################################
def clean_caption(caption):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # tokenize
    desc = caption.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word)>1]
    # remove tokens with numbers in them
    desc = [word for word in desc if word.isalpha()]
    # store as string
    #caption = settings.start_seq +" "+ ' '.join(desc) +" "+ settings.end_seq
    return ' '.join(desc)

def final_caption(caption):
    desc = settings.start_seq +" "+ clean_caption(caption) +" "+ settings.end_seq
    return desc
