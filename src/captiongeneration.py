
from pathlib import Path
import h5py
import numpy as np
import logging
import os
import pickle

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

import GloVepreprocessing
import storage
import model
import logging

logger = logging.getLogger()

def caption_generation(set_name, index, epochs = 10):
    
    preprocessor = None
    logger.info("loading preprocessor")
    try:
        with open("../data/preprocessor.pickle", 'rb') as handle:
            preprocessor = pickle.load(handle)
    except FileNotFoundError:
        logger.info("file not found, initilizing GloVe")      
        preprocessor = GloVepreprocessing.GloVepreprocessor()
        with open("../data/preprocessor.pickle", 'wb') as handle:
            logger.info("serializing preprocessor")
            pickle.dump(preprocessor, handle)
     
    logger.info("reading data for caption generation")
    status, image, caption = storage.read_image(set_name, index)
    if status != 200:
        return "None"
    
    _ ,inference_initialiser_model,inference_model = model.ShowAndTell(preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.VOCAB_SIZE, preprocessor.EMBEDDING_SIZE, 60, preprocessor.weights)
            
    logger.info("loading inference init model") 
    #inference_initialiser_model = load_model("../data/models/inference_init{0}.saved".format(epochs))
    inference_initialiser_model.load_weights("../data/models/w_inference_init{0}.saved".format(epochs))
    
    #training_model = keras.models.load_model("../data/models/train_{0}.saved".format(epochs))
    logger.info("loading inference model")
    #inference_model = load_model("../data/models/inference_{0}.saved".format(epochs))
    inference_model.load_weights("../data/models/w_inference_{0}.saved".format(epochs))
    
    a0 = np.zeros((1, 60))
    c0 = np.zeros((1, 60))
    image = K.expand_dims(image, axis=0)

    state_a, state_c = inference_initialiser_model.predict({'image_input':image, 'a0':a0, 'c0':c0})

    generated_caption = ["[CLS]"]   
    current_word = None
    
    for t in range(preprocessor.MAX_SEQUENCE_LENGTH):
        embedded_generated_caption = preprocessor.GloVe_embed_tokens(generated_caption, preprocessor.weights)
        logger.debug("-------------Generated Caption---------------: " + " ".join(generated_caption))

        output, state_a, state_c = inference_model.predict({"caption_input":np.reshape(embedded_generated_caption,(1, preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.EMBEDDING_SIZE)), "a1":state_a, "c1":state_c})
        
        
        #print("Output shape"+str(output.shape)+"Starting Word research")
        #temp_caption = []
        #for i in range(preprocessor.MAX_SEQUENCE_LENGTH):
            #print("Outputoutput[0, i]"+str(len(output[0, i])))
            #generated_word2 = np.argmax(output[0, i])
            #print(generated_word2)
            #print(preprocessor.idx2word[generated_word2])
            #temp_caption.append(preprocessor.idx2word[generated_word2])
        #print(" ".join(temp_caption))
        
        generated_word = np.argmax(output[0, t])
        logger.debug("Generated Word Index" + str(generated_word))
        logger.debug("Generated Word" + preprocessor.idx2word[generated_word])
        print("Generated Word Index" + str(generated_word))
        print("Generated Word" + preprocessor.idx2word[generated_word])

        generated_caption.append(preprocessor.idx2word[generated_word])

        if generated_word == preprocessor.word2idx["[SEP]"]:
            break
     
    candidate_caption = generated_caption[1:]
    logger.info("Candidate Caption: = "+" ".join(candidate_caption))
    return " ".join(candidate_caption), caption, image
