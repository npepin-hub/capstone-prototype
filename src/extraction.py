import pandas as pd
import requests
from PIL import Image
import logging
import io
import os
from pathlib import Path
import h5py
import numpy as np
import storage

def request_data_and_store(dataframe, size, set_name, batch_size = 5):
    """ Given a panda dataframe containing a list of urls and their corresponding caption, call 
        the urls and stores each thumbnailed-padded-to-size Image/Caption into a single hdf5 file.
        Parameters:
        ---------------
        dataframe    the pandas dataframe raw dataset
        size         the targeted size of the image in the form of a tuple (height, width)
        batch_size   future batch size...
        set_name     validation or training 

        Returns:     
        ----------
        nothing for now...
    """
    logger = logging.getLogger()

    # TODO - work with batches - For now, we will work on the batch_size-th first data of the validation dataset
    for index, row in dataframe.iterrows():
        if not(storage.exist(set_name ,index)):

            # Gets URLs
            r = requests.get(row.url)
            logger.info(set_name+"-- URL#"+str(index)+" Http code: "+str(r.status_code))
            if (r.status_code == 200):
                logger.debug(row.url)
                img = Image.open(io.BytesIO(r.content))
                img.thumbnail(size, Image.ANTIALIAS)

                padded_image = Image.new("RGB", size)
                padded_image.paste(img, (int((size[0] - img.size[0])/2), int((size[1] - img.size[1])/2)))
                print(row.caption)
                #plt.imshow(np.asarray(padded_image))
                #plt.show()
                # Stores the image into a hdf5 file

                storage.store_image(set_name, index, padded_image, row.caption)
                
            else:
                storage.store_status(set_name, index, str(r.status_code))
                
            dataframe.at[index, "statuscode"] = int(r.status_code)

    return