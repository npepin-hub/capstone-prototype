from config import settings
from filelock import FileLock
import h5py
import io
import logging
import numpy as np
import os
from os import walk
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
import yaml



#####################################################################################################################
#   This file provides functions that store and access the raw data that will be used for training                  #
#   and validating our model.                                                                                       #
#   We will be using a couple of .h5 files to store all the 3.3M images and their captions.                         #
#####################################################################################################################

    
##########################
#    Helper functions    #
##########################

def get_file_size():
    """
    Returns the number of images per h5 files
    """
    return settings.images_per_storage

def get_path(set_name ,file_number): 
    """
    Given a file number and the type of data we are looking for (train or validate), returns the path to the given file and its associated .lock file
    Parameters:
    ----------
    set_name      "validate" or "train"
    file_number    the file number to consider

    Returns:
    ----------
    file_path      path to the file
    lock_path      path to its .lock file
    """
    
    dirname = os.path.dirname(__file__)
    images_path = dirname+"/"+settings.data_path
    file_path = Path(images_path+set_name+'_'+str(file_number)+'.h5')
    lock_path = Path(images_path+set_name+'_'+str(file_number)+'.h5.lock')   

    return file_path, lock_path


def get_file_number(index): 
    """
    Return the .h5 file nb containing a given image index
    """
    file_nb = int(int(index) / get_file_size())     
    return file_nb


def get_file_path_from_idx(set_name ,index):    
    """
    Returns the path to an .h5 image file given an index of an image it contains
    """
    file_number = get_file_number(index)
    return get_path(set_name ,file_number)


def get_file_numbers(start_index, stop_index): 
    """
    Returns the range of file numbers in which a range of image indexes is
    """
    start, stop = get_file_number(start_index),get_file_number(stop_index)     
    return [start, stop]


def exist(set_name ,index):
    """
    Check if a given index already exists in a given set (validate or train) of .h5 file
    """
    logger = logging.getLogger()

    file_path, lock_path = get_file_path_from_idx(set_name ,index)
    # Locks the file since it will be accessed from multiple threads
    
    
    lock = FileLock(lock_path)
    with lock.acquire():
        try:
            with h5py.File(file_path , "r") as file:
                if (str(index) in file):
                    return True
        except OSError as e:
            logger.info("storage.exist function: "+set_name+"-- Index# "+str(index)+" Error: "+str(e))
        return False
    
    
  # lock = FileLock(lock_path)
  #  lock.acquire()  
  #  file = None
  #  try:      
  #      file = h5py.File(file_path , "r")
  #      if (str(index) in file):
  #          return True
  #  except OSError as e:
  #      logger.info("storage.exist function: "+set_name+"-- Index# "+str(index)+" Error: "+str(e))
  #  finally:
  #      if file != None:
  #          file.close()
  #      lock.release()
  #  return False
                     
##############################################################################                    
#     Read/write (image/caption/http request status code) into .h5 files     #
##############################################################################
 
def store_status(set_name, index, status_code):
    """ Stores an http-request status code under the index in a given set_name .h5 storage
    Used to store the HTTP status code only when different from 200 (ie: no picture retrieved)
    Parameters:
    ---------------
    set_name    validate or train
    index       integer unique ID for image 
    status_code http-request status code           
    """
    logger = logging.getLogger()
    logger.info(set_name+"store_status.exist:-- URL#"+str(index)+" Storing status--- "+str(status_code))
    
    file_path, lock_path = get_file_path_from_idx(set_name ,index)
    # Locks the file since it will be accessed from multiple threads

    lock = FileLock(lock_path)
    lock.acquire()  
    file = None
    try:                
        # Create/retrieve a new HDF5 file
        file = h5py.File(file_path , "a") 
        
        # Remove image_id from file if already exists
        if (str(index) in file):
            del file[str(index)]

        # Create a group/dataset in the file for a given image/caption
        group = file.create_group(str(index))
        meta_set = group.create_dataset(
            "status", data=status_code
        )
    except (OSError, Exception) as e:
        logger.info("store_status: "+set_name+"-- URL#"+str(index)+" Error: "+str(e))
    finally:
        if file != None:
            file.close()
        lock.release()


def store_image(set_name, index, image, features, caption):
    """ 
    Stores a single image/caption under the index in a given set_name .h5 storage
    Parameters:
    ---------------
    set_name    validate or train
    index       integer unique ID for image 
    image       image to be stored   
    caption     image caption        
    """
    logger = logging.getLogger()
    logger.info("storage.store_image"+set_name+"-- INDEX#"+str(index)+" STORING--- ")

    file_path, lock_path = get_file_path_from_idx(set_name ,index)
                     
    # Locks the file since it will be accessed from multiple threads
    
    lock = FileLock(lock_path)
    with lock.acquire():
        try:
            with h5py.File(file_path , "a") as file:        
                # Remove index from file if already exists
                if (str(index) in file):
                    logger.debug("storage.store_image"+set_name+"-- INDEX#"+str(index)+" DELETING--- ")
                    del file[str(index)]
                logger.debug("storage.store_image"+set_name+"-- INDEX#"+str(index)+" GROUP--- ")
                # Create a group/dataset in the file for a given image/caption
                group = file.create_group(str(index))
                if not (image is None):
                    group.create_dataset(
                        "image", np.shape(image), h5py.h5t.STD_U8BE, data=np.asarray(image)
                    )
                if not (features is None):
                    group.create_dataset(
                        "features", np.shape(features), h5py.h5t.IEEE_F32BE, data=np.asarray(features)
                    ) 
                group.create_dataset(
                    "caption", data=caption
                )
                group.create_dataset(
                    "status", data="200"
                )
                logger.info("storage.store_image"+set_name+"-- URL#"+str(index)+" IMAGE STORED--- ")
        except (OSError, Exception) as e:
            logger.warn("storage.store_image"+set_name+"-- URL#"+str(index)+" Error: "+str(e))



def read_image(set_name, index):
    """ 
    Reads a single Image/Caption given its index out of a set_name .h5 storage
    Parameters:
    ---------------
    set_name    validate or train
    index       integer unique ID for image

    Returns:
    ----------
    status      http status code
    image       image array stored
    caption     associated caption
    """
    logger = logging.getLogger()
    # Gets the file number this index is in
    file_nb = get_file_number(index)
    # Open the HDF5 file
    file_path , lock_path = get_path(set_name ,file_nb)
    
    
    lock = FileLock(lock_path)
    with lock.acquire():
        try:
            with h5py.File(file_path , "r") as file:
                group = file[str(index)]
                status = np.array(group["status"]).astype("int32")
                if (status == 200):
                    logger.debug("read_image function found: "+set_name+"-- Index# "+str(index)+" Status: "+str(status))
                    features = None
                    image = None
                    if group.get("features") is not None:
                        logger.debug("storage.read_image: Get features")
                        features = np.array(group["features"]).astype("float32")
                    if group.get("image") is not None:
                        logger.debug("storage.read_image: Get image")
                        image = np.array(group["image"]).astype("uint8")
                    caption = np.array(group["caption"]).astype("str")
                    return status, image, features, caption
                else:
                    logger.debug("read_image function: "+set_name+"-- Index# "+str(index)+" Status: "+str(status))
                    return status, None, None, None
        except (OSError, Exception) as e:
            logger.debug("read_image function: "+set_name+"-- Index# "+str(index)+" Error: "+str(e))
        logger.debug("read_image function: "+set_name+"-- Index# "+str(index))
    return 500, None, None, None

 
def get_last_stored_index(set_name):
    """ 
    Images and captions are stored in various h5 files.
    Gets to the last existing h5 files before adding to it - Used when resuming the load of data after interruption.
    Parameters:
    ---------------
    set_name   validate or train

    Returns:
    ----------
    index      the last index of the last .h5 file
    """ 
    logger = logging.getLogger()
    images_path = settings.data_path
    
    f = []
    for (dirpath, dirnames, filenames) in walk(images_path):
        f.extend(filenames)
        break

    filtered_list = list(filter(lambda i: (i.endswith(".h5") and i.find(set_name+"_") == 0), f))
    logger.info("storage.get_last_stored_index --filtered_list.size-- " + str(len(filtered_list)))
    logger.info(filtered_list)
    if len(filtered_list) == 0:
        return 0
    
    #Get the last image index in the last file
    last_file_idx = str(len(filtered_list) -1)
    file = h5py.File(Path(images_path+set_name+"_"+last_file_idx+".h5") , "r")
    keys_list = list(file.keys())
    logger.info("storage.get_last_stored_index --KeysList size-- " + str(len(keys_list)))
    next_index = len(keys_list) + get_file_size() * int(last_file_idx)
    return next_index - 1

