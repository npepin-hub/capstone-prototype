from config import settings
import pandas as pd
import gc
from guppy import hpy
import logging
import logging.config
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import yaml
import sys

from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import extraction
import model


##############################################################################################################
#  This file is used to extract the images from the raw data (url) and store image/caption in .h5 files.     #
##############################################################################################################

def extract_data(set_name, img_size, start_index=0, extraction_size=100000):
# Extracts image and caption from tsv and store padded(image),caption and HTTPStatus code
# into a train or validation hdf5 file
    logger = logging.getLogger()
    data_df = None
    if "train" in set_name:
        # Read TSV file
        train_rawdata_file_path = settings.train_raw_data
        data_df = pd.read_table(train_rawdata_file_path, header = None, names = ['caption', 'url'] )
        
    elif "validate" in set_name:
        # Read TSV file
        validation_rawdata_file_path = settings.validation_raw_data
        data_df = pd.read_table(validation_rawdata_file_path, header = None, names = ['caption', 'url']) 

    else:
        logger.info("loader.extract_data: get your set_name right! What is it you want to load again?")
        return
    
    # Extract images and stores images/captions
    extraction.request_data_and_store(data_df, img_size, set_name, start_index, extraction_size)
    logger.info("loader.extract_data: all the data have been extracted and stored.")

if __name__ == "__main__":   
    logger = logging.getLogger()
    
    img_size = sys.argv[2].split(',')
    img_size = tuple(map(int, img_size)) 
    
    logger.info("Loader.extract_data: Starting extraction")
    extract_data(sys.argv[1], img_size, int(sys.argv[3]), int(sys.argv[4]))
    logger.info("Loader.extract_data: End of extraction")
    

    


    
