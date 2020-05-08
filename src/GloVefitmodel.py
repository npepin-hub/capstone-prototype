from matplotlib import pyplot as plt
import logging
import numpy as np
import os
import pickle
import captiongeneration


from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras.optimizers import Adam

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


# Loads model and weights
training_model, inference_initialiser_model, inference_model = model.ShowAndTell(preprocessor.MAX_SEQUENCE_LENGTH, preprocessor.VOCAB_SIZE, preprocessor.EMBEDDING_SIZE, 60, preprocessor.weights)

loss_function = preprocessor.get_loss_function()
training_model.compile(loss=loss_function, optimizer=Adam(lr = 0.03), metrics=['accuracy'])

checkpoint_path = "../data/models/chk/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=2)

tb_callback = TensorBoard(log_dir='/var/log', histogram_freq=1, write_graph=False, write_grads=False, write_images=False, update_freq='batch', embeddings_freq=0)

batch_size = 10
epochs=100  
    
history = training_model.fit(preprocessor.generator('train', batch_size=batch_size), steps_per_epoch=20, epochs=epochs, verbose=1, callbacks=[cp_callback])

training_model.save_weights("../data/models/w_train_{0}.saved".format(epochs))
inference_initialiser_model.save_weights("../data/models/w_inference_init{0}.saved".format(epochs))
inference_model.save_weights("../data/models/w_inference_{0}.saved".format(epochs))


for key in history.history.keys():
    f = plt.figure()
    data = history.history[key]
    plt.plot(data)
plt.show()
    
