from config import settings
import h5py
import io
import logging
import numpy as np
import os
import pandas as pd
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

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk

import GloVepreprocessing
import storage
import model
import predict

# evaluate the skill of the model
def evaluate_model(set_name, inference_model, preprocessor, max_size):
    # Max_size = validation set size
    actual, predicted = list(), list()
    # step over the validation set
    
    idx = 0
    missed_idx = 0

    while (idx + missed_idx < max_size):
        try:
            logger.debug("Loading features for evaluation " + str(idx + missed_idx))
            status,_,features,caption = storage.read_image(set_name, idx + missed_idx)                        
            caption = str(caption).split()[1:-1]

            if (int(status) == 200):
                # generate description
                yhat = predict.generate_description(inference_model, features, preprocessor)
                actual.append([caption])
                predicted.append(yhat.split())
                print("Predicted: "+yhat+ "- Actual: "+" ".join(caption))
                idx = idx + 1
            else:
                missed_idx = missed_idx + 1
        except KeyError:
            # Ignores files not found - probably an HHTP error when requesting the URL
            missed_idx = missed_idx + 1
            continue

    # calculate BLEU score
    cc = SmoothingFunction()
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=cc.method3))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method3))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=cc.method3))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method3))



if __name__ == "__main__":
 
    # Get logger
    logger = logging.getLogger()   
    
    # Get embedding matrix
    preprocessor = GloVepreprocessing.preprocessor_factory()

    # get validation set size
    validation_rawdata_file_path = settings.validation_raw_data
    data_df = pd.read_table(validation_rawdata_file_path, header = None, names = ['caption', 'url']) 
    max_size = data_df["caption"].count()
    #print(max_size)

    set_name = "features/resnet_validate"
    epoch = 42

    features_model = predict.get_features_model()
    inference_model = predict.get_inference_model(epoch, preprocessor)
 
    evaluate_model(set_name, inference_model, preprocessor, max_size)




