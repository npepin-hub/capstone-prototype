from config import settings
import logging
import logging.config
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import GloVepreprocessing
import captiongeneration
import model


preprocessor = None
logger = logging.getLogger()

try:
    with open(settings.glove_embed_data, 'rb') as handle:
        preprocessor = pickle.load(handle)
except FileNotFoundError:      
    preprocessor = GloVepreprocessing.GloVepreprocessor()
    with open(settings.glove_embed_data, 'wb') as handle:
        print("before pickle dump")
        pickle.dump(preprocessor, handle)


# Loads model and weights
model = model.injectAndMerge(preprocessor)

# Retrieve custom loss function
#loss_function = preprocessor.get_loss_function()

# Compile model
#training_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr = 0.001), metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr = 0.001))
print(model.summary())


checkpoint_path = "../models/chk/"
checkpoint_dir = os.path.dirname(checkpoint_path)

initial_epoch = 1
batch_size = 5
epochs=2
steps_per_epoch = 1000

try:
    logger.info("Re-loading training model to train further") 
    print("Re-loading training model to train further")
    model.load_weights(f"../models/w_train_{initial_epoch}.saved")

except NotFoundError:
    print("No training model to load - starting fresh")
    logger.info("No training model to load - starting fresh")
    initial_epoch = 0
    
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path + 'wtrain-{epoch:03d}',save_weights_only=True,verbose=2, save_best_only=False, mode='auto', monitor='loss', save_freq='epoch')

tb_callback = TensorBoard(log_dir='/var/log/TensorBoard', histogram_freq=1, write_graph=False, write_grads=False, write_images=False, update_freq='batch', embeddings_freq=0)

generator = preprocessor.generator('resnet_train', batch_size=batch_size, start_index=0)
history = model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch = initial_epoch, verbose=1, callbacks=[cp_callback, tb_callback], workers=1)

model.save_weights(f"../models/w_train_{epochs}.saved")


"""for key in history.history.keys():
    f = plt.figure()
    data = history.history[key]
    plt.plot(data)
plt.show()"""
    
