import pandas as pd
import requests
import PIL
from PIL import Image
import time
import logging
import io
import os
from pathlib import Path
import h5py
import numpy as np
import concurrent.futures
from threading import BoundedSemaphore
from itertools import islice
import storage


def get_image(index, url, size, set_name):

    logger = logging.getLogger()
    logger.info(set_name+"-- Fetching URL#"+str(index))

    # Gets URLs
    try:
        r = requests.get(url)

        logger.info(set_name+"-- URL#"+str(index)+" Http code: "+str(r.status_code))
        if (r.status_code == 200):
            logger.info(url)
            img = Image.open(io.BytesIO(r.content))
            img.thumbnail(size, Image.ANTIALIAS)

            padded_image = Image.new("RGB", size)
            padded_image.paste(img, (int((size[0] - img.size[0])/2), int((size[1] - img.size[1])/2)))

            #plt.imshow(np.asarray(padded_image))
            #plt.show()

            return int(r.status_code), padded_image
        else:
            return int(r.status_code), None

    except (requests.exceptions.ConnectionError, requests.exceptions.InvalidURL, requests.exceptions.SSLError, requests.exceptions.ContentDecodingError) as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))

    except PIL.UnidentifiedImageError as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))

    except OSError as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))
        
    except Exception as e:
        logger.info(set_name+"-- URL#"+str(index)+" Http error: "+str(e))


    return 500, None    



def store_handler(set_name, index, caption, status_code, padded_image=None):
    logger = logging.getLogger()
    logger.info(set_name+"--Handler URL#"+str(index)+" Http code: "+str(status_code))

    if (int(status_code) == 200):
        storage.store_image(set_name, index, padded_image, caption)            
    else:
        storage.store_status(set_name, index, str(status_code)) 
    return 



def request_data_and_store(dataframe, size, set_name, start_index = 0):
    """ Given a panda dataframe containing a list of urls and their corresponding caption, get 
        the images and stores each thumbnailed-padded-to-size Image/Caption into a single hdf5 file.
        Parameters:
        ---------------
        dataframe    the pandas raw dataset
        size         the targeted size of the image in the form of a tuple (height, width)
        set_name     validation or training
        start_index  the index at which the insertion will start

        Returns:     
        ----------
        nothing for now...
    """
    logger = logging.getLogger()
   
    queue = BoundedSemaphore(100)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:          
        for index, row in islice(dataframe.iterrows(), start_index, None):                       
            if (index % 1000) == 0:
                logger.info("--Processing " + str(index))

            if not(storage.exist(set_name ,index)):

                response_handler =  \
                    lambda future, index=index, caption=row.caption: store_handler(set_name, index, caption, future.result()[0], future.result()[1])

                release_handler = lambda future: queue.release()
                
                logger.info("-- Before ACQUIRE -- " + str(index))
                queue.acquire()
                
                logger.info("-- Before SUBMIT -- " + str(index))
                future_image = executor.submit(get_image, index, row.url, size, set_name)
                future_image.add_done_callback(response_handler)
                future_image.add_done_callback(release_handler)

    return
