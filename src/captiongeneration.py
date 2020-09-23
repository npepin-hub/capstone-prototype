
from pathlib import Path
import h5py
import numpy as np
import logging
import os
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
import logging

logger = logging.getLogger()

def caption_generation(set_name, index, epoch = 10):
    
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
        return "No image", caption, None
    
    img = Image.fromarray(image)
    img = img.resize(size=(224, 224))
    image_batch = np.expand_dims(img, axis=0)
    image_batch = preprocess_input(image_batch)
    image = image_batch[0]
    
    
    checkpoint_path = "../data/models/chk/"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    training_model,inference_initialiser_model,inference_model = model.ShowAndTell(preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.VOCAB_SIZE, 256, 60, preprocessor.weights)
    
    logger.info("loading training  model") 
    #training_model.load_weights(f"../data/models/w_train_{epoch}.saved")
    training_model.load_weights(f"{checkpoint_path}wtrain-{epoch:03d}")

    logger.info("loading inference init model") 
    #inference_initialiser_model = load_model("../data/models/inference_init{0}.saved".format(epochs))
    #inference_initialiser_model.load_weights("../data/models/w_inference_init{0}.saved".format(epoch))
    #inference_initialiser_model.load_weights(f"../data/models/chk/wtrain-{epoch}", by_name=True, skip_mismatch=True)
    
    logger.info("loading inference model")
    #inference_model = load_model("../data/models/inference_{0}.saved".format(epochs))
    #inference_model.load_weights("../data/models/w_inference_{0}.saved".format(epoch))
    #inference_model.load_weights(f"../data/models/chk/wtrain-{epoch}", by_name=True, skip_mismatch=True)
    
    
    a0 = np.zeros((1, 60))
    c0 = np.zeros((1, 60))

    #state_a, state_c = inference_initialiser_model.predict({'image_input':image, 'a0':a0, 'c0':c0})

    generated_caption = ["startseq"]   
    current_word = None
    
    for t in range(preprocessor.MAX_SEQUENCE_LENGTH-1):
        
        #embedded_generated_caption = preprocessor.GloVe_embed(preprocessor.get_sequences_ids([generated_caption]))
        generated_caption_sequence_ids = preprocessor.get_sequences_ids([generated_caption])
        #logger.debug("-------------Generated Caption---------------: " + " ".join(generated_caption))

        #output, state_a, state_c = inference_model.predict({"caption_input":np.reshape(embedded_generated_caption,(1, preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.EMBEDDING_SIZE)), "a1":state_a, "c1":state_c})
        #output = training_model.predict({"image_input":image_batch, "caption_input":np.reshape(embedded_generated_caption,(1, preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.EMBEDDING_SIZE))})
        output = training_model.predict({"image_input":image_batch, "caption_input":np.reshape(generated_caption_sequence_ids,(1, preprocessor.MAX_SEQUENCE_LENGTH))})
        print("OUTPUT SHAPE"+ str(output.shape))
        print(np.argmax(output[0]))
        #print("OUTPUT"+preprocessor.idx2word[ np.argmax(output[0])])
        
        
        #print("Output shape"+str(output.shape)+"Starting Word research")
        #temp_caption = []
        #for i in range(preprocessor.MAX_SEQUENCE_LENGTH):
            #print("Outputoutput[0, i]"+str(len(output[0, i])))
            #generated_word2 = np.argmax(output[0, i])
            #print(generated_word2)
            #print(preprocessor.idx2word[generated_word2])
            #temp_caption.append(preprocessor.idx2word[generated_word2])
        #print(" ".join(temp_caption))
        
        generated_word = np.argmax(output[0])
        logger.debug("Generated Word Index" + str(generated_word))
        logger.debug("Generated Word" + preprocessor.idx2word[generated_word])
        print("Generated Word Index" + str(generated_word))
        print("Generated Word" + preprocessor.idx2word[generated_word])

        generated_caption.append(preprocessor.idx2word[generated_word])

        if generated_word == preprocessor.word2idx["endseq"]:
            print("we have reached the end of the sentence")
            break
     
    candidate_caption = generated_caption[1:]
    logger.info("Candidate Caption: = "+" ".join(candidate_caption))
    return " ".join(candidate_caption), caption, image


# Test the ResNet50 classification) - 1st part of the model
def classification_generation(set_name, index, epoch = 10):
    
    #image_shape = (300, 300, 3)
    #X_input = Input(image_shape, name="image_input")  
    #model = ResNet50(weights='imagenet',include_top=False, input_tensor=X_input, input_shape=image_shape)

    model = ResNet50(weights='imagenet')
    #print(model.summary())
    print('Model loaded. Started serving...')
    print(index)
    #image = tf.keras.preprocessing.image.load_img("../data/panda.png", target_size = (300,300))
    #status = 200
    #caption = "a super duper panda"
    #size = (300,300)
    #img = Image.open("../data/fennec.jpg")
    #img.thumbnail(size, Image.ANTIALIAS)
    
    #padded_image = Image.new("RGB", size)
    #padded_image.paste(img, (int((size[0] - img.size[0])/2), int((size[1] - img.size[1])/2)))

    
    #storage.store_image(set_name, index, padded_image, caption)

    status, image, caption = storage.read_image(set_name, index)
    
    if status != 200:
        return "No image", caption, None
    print(caption)
    #image_batch = np.expand_dims(image, axis=0)

    
    #image_batch = tf.image.resize(
    #    image_batch, [224, 224], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True,
    #    antialias=False, name=None
    #)
    
    img = Image.fromarray(image)
    img = img.resize(size=(224, 224))
    
    image_batch = np.expand_dims(img, axis=0)


    # Preprocessing the image
    print('PIL image size = ', image_batch.shape)


    
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    
 

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #image_batch = preprocess_input(image_batch, mode='caffe')
    image_batch = preprocess_input(image_batch)
    
    #with graph.as_default():    
        
    preds = model.predict(image_batch)
    # pred_class = preds.argmax(axis=-1)            # Simple argmax
    pred_class = decode_predictions(preds, top=5)# ImageNet Decode
    print (pred_class)
    result = str(pred_class[0][0][1]) + " " +   str(pred_class[0][1][1]) + " " + str(pred_class[0][2][1])            # Convert to string
        
    return result, caption, image_batch[0]


