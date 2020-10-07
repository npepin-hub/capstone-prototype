from config import settings
import gc
from guppy import hpy
import logging
import logging.config
import numpy as np
import os
import pickle
import sys
import yaml


from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import GloVepreprocessing
import model

##############################################################################
#  This file is used to launch the compilation and training of our model     #
##############################################################################

def setupLogging():
    log_file_path = settings.log_config 
    with open(log_file_path, 'rt') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
        
def fit(set_name, initial_epoch, batch_size, epochs, steps_per_epoch, start_index):
    logger = logging.getLogger()
    # Get embedding matrix
    preprocessor = None
    try:
        with open(settings.glove_embed_data, 'rb') as handle:
            preprocessor = pickle.load(handle)
    except FileNotFoundError:      
        preprocessor = GloVepreprocessing.GloVepreprocessor()
        with open(settings.glove_embed_data, 'wb') as handle:
            logger.info("Saving new pickle")
            pickle.dump(preprocessor, handle)

    # Loads model
    training_model = model.injectAndMerge(preprocessor)

    # Retrieve custom loss function
    #loss_function = preprocessor.get_loss_function()

    # Compile model
    training_model.compile(loss=settings.LOSS, optimizer=Adam(lr = settings.LEARNING_RATE))
    logger.info(training_model.summary())
    
    checkpoint_path = settings.model_path+"chk/"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    try:        
        logger.info("Re-loading training model to train further") 
        training_model.load_weights(f"{checkpoint_path}wtrain-{initial_epoch:03d}")
    except NotFoundError:
        logger.info("No training model to load - starting fresh")
        initial_epoch = 0

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path + 'wtrain-{epoch:03d}',save_weights_only=True,verbose=2, save_best_only=False, mode='auto', monitor='loss', save_freq='epoch')

    # Tensorboard callback
    #tb_callback = TensorBoard(log_dir='/var/log/TensorBoard', histogram_freq=1, write_graph=False, write_grads=False, write_images=False, update_freq='batch', embeddings_freq=0)

    tb_callback = TensorBoard(log_dir=settings.tensorboard_log_dir)        
    
    for i in range(0,1):
        logger.info("Training Iteration " + str(i)) 

        # create the data generator
        training_model.fit(preprocessor.generator(set_name, batch_size=batch_size, start_index=start_index), steps_per_epoch=steps_per_epoch, epochs=epochs + i * epochs, initial_epoch = initial_epoch + i * epochs, verbose=1, callbacks=[cp_callback,tb_callback], workers=1)
    
        training_model.save(settings.model_path+"w_train_iteration%d.h5" %i)
        h = hpy()
        heap = h.heap()
        if heap is not None:
            logger.info(heap)
        gc.collect()
        heap = h.heap()
        if heap is not None:
            logger.info(heap)
        
if __name__ == "__main__":
    # Get logger
    setupLogging()
    logger = logging.getLogger()
    
    logger.info("train.fit: Starting training the model")
    fit(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

    logger.info("End of training of the model.")
