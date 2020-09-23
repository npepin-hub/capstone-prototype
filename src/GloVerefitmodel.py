from matplotlib import pyplot as plt
import logging
import logging.config
import numpy as np
import os
import pickle
import captiongeneration


from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import GloVepreprocessing
import model

preprocessor = None

try:
    with open("../data/preprocessor.pickle", 'rb') as handle:
        preprocessor = pickle.load(handle)
except FileNotFoundError:      
    preprocessor = GloVepreprocessing.GloVepreprocessor()
    with open("../data/preprocessor.pickle", 'wb') as handle:
        print("before pickle dump")
        pickle.dump(preprocessor, handle)

initial_epoch = 36
batch_size = 1
epochs=initial_epoch + 400
steps_per_epoch = 50

checkpoint_path = "../models/chk/"
checkpoint_dir = os.path.dirname(checkpoint_path)

logger = logging.getLogger()

# Loads model and weights
training_model, inference_initialiser_model, inference_model = model.ShowAndTell(preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.VOCAB_SIZE, preprocessor.EMBEDDING_SIZE, 60, preprocessor.weights)

#loss_function = preprocessor.get_loss_function()
training_model.compile(loss=loss_function, optimizer=Adam(lr = 0.000ßΩ1), metrics=['accuracy'])
#custom_objects={"loss":loss_function}

logger.info("Re-loading training model to train further") 
print("Re-loading training model to train further")

training_model.load_weights(f"{checkpoint_path}wtrain-{initial_epoch:03d}")
#training_model.load_weights(f"../models/w_train_{initial_epoch}.saved")
  
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path + 'wtrain-{epoch:03d}',save_weights_only=True,verbose=2, save_best_only=False, mode='auto', monitor='loss', period=2)

tb_callback = TensorBoard(log_dir='/var/log/TensorBoard/', histogram_freq=1, write_graph=True, write_grads=False, write_images=False, update_freq='batch', embeddings_freq=0)
    
history = training_model.fit(preprocessor.generator('train', batch_size=batch_size, start_index=3600), steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch = initial_epoch, verbose=2, callbacks=[cp_callback,tb_callback])

training_model.save_weights(f"../models/w_train_{epochs}.saved")
inference_initialiser_model.save_weights(f"../models/w_inference_init{epochs}.saved")
inference_model.save_weights(f"../models/w_inference_{epochs}.saved")

for key in history.history.keys():
    f = plt.figure()
    data = history.history[key]
    plt.plot(data)
plt.show()
    
