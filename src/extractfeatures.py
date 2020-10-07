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

import GloVepreprocessing
import captiongeneration
import extraction
import model


################################################################################################################################
#  This file is used to pre-extract the features from our training set and store them in the h5 files that will be fed to the  #  #  model when training                                                                                                         #
################################################################################################################################


# Get logger
def setupLogging():
    log_file_path = settings.log_config
    with open(log_file_path, 'rt') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)


def extract_features(set_name, start_index=0, extraction_size=100000):
# Extracts the features of given images and store them into a train or validation hdf5 file
    logger = logging.getLogger()
    data_df = None
    
    # Extract image features and stores features/captions in h5 files
    extraction.extract_resnet_features_and_store(set_name, start_index, start_index + extraction_size)
    logger.info("extractfeatures.extract_features: all the features have been extracted and stored.")

if __name__ == "__main__":
    
    setupLogging()
    logger = logging.getLogger()
    
    logger.info("extractfeatures.extract_features: Starting extraction")
    extract_features(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    logger.info("extractfeatures.extract_features: End of extraction")
    

    


    



    
