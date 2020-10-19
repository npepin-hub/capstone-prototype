import argparse
from config import settings
import gc
from guppy import hpy
import logging
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
        
def train(set_name, initial_epoch, batch_size, epochs, steps_per_epoch, start_index, learning_rate, loss, optimizer):
    logger = logging.getLogger()
 
    # Get embedding matrix
    preprocessor = GloVepreprocessing.preprocessor_factory()

    # Loads model
    training_model = model.injectAndMerge(preprocessor)

    # Retrieve custom loss function
    #loss_function = preprocessor.get_loss_function()

    # Compile model
    training_model.compile(loss=loss, optimizer=Adam(lr = learning_rate))
    logger.info(training_model.summary())
    
    checkpoint_path = settings.model_path+"chk/"
    refit_path = settings.refit_path
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    try:        
        logger.info("Re-loading training model to train further" + f"{refit_path}wtrain-{initial_epoch:03d}") 
        training_model.load_weights(f"{refit_path}wtrain-{initial_epoch:03d}")
    except NotFoundError:
        logger.info("No training model to load - starting fresh")
        initial_epoch = 0
        exit(1)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path + 'wtrain-{epoch:03d}',save_weights_only=True,verbose=1, save_best_only=False, mode='auto', monitor='loss', save_freq='epoch')

    # Tensorboard callback
    #tb_callback = TensorBoard(log_dir='/var/log/TensorBoard', histogram_freq=1, write_graph=False, write_grads=False, write_images=False, update_freq='batch', embeddings_freq=0)

    #tb_callback = TensorBoard(log_dir=settings.tensorboard_log_dir)        
    steps = 2
    for e in range(initial_epoch,epochs, steps):
        logger.info("Epoch: " + str(e)) 
        for i in range(0,steps):
            logger.info("Iteration: " + str(i)) 
            # create the data generator
            training_model.fit(preprocessor.generator(set_name, batch_size=batch_size, start_index=start_index), steps_per_epoch=steps_per_epoch, epochs=e + i + 1, initial_epoch = e + i, verbose=1, callbacks=[cp_callback], workers=1)
        start_index += batch_size * steps_per_epoch
    
    training_model.save(settings.model_path+f"wtrain-{initial_epoch:03d}")

# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    # Batch size and set_name
    parser.add_argument('--set_name', type=str, default="features/resnet_train")
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)    
    parser.add_argument('--steps_per_epoch', type=int, default=500)    
    parser.add_argument('--start_index', type=int, default=0)    
    
    parser.add_argument('--learning_rate', type=float, default=settings.LEARNING_RATE)
    parser.add_argument('--loss', type=str, default=settings.LOSS)
    parser.add_argument('--optimizer', type=str, default="adam")

    parser.add_argument('--tensorboard_log_dir', type=str, default='/opt/ml/output/logs/tensorboard')

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=float, default=100)
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    #parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()
       
        
if __name__ == "__main__":
    args = parse_args()

    print(args.initial_epoch)
    #exit

    # Get logger
    logger = logging.getLogger()
    logger.info("train.fit: Starting training the model")
    train(args.set_name, args.initial_epoch, args.batch_size, args.epochs, args.steps_per_epoch, args.start_index, args.learning_rate, args.loss, args.optimizer)
    #fit(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

    logger.info("End of training of the model.")
