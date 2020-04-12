import pandas as pd
import requests
from PIL import Image
import io
import os
from os import walk
from pathlib import Path
import h5py
import numpy as np
import logging
from filelock import FileLock

def get_file_size():    
    K_FOLD = 100000
    return K_FOLD

def get_path(set_name ,index):    
    file_number = get_file_number(index)
    return get_path(set_name ,file_number)

def get_path(set_name ,file_number):    
    file_path = Path('../data/img/'+set_name+'_'+str(file_number)+'.h5')
    lock_path = Path('../data/img/'+set_name+'_'+str(file_number)+'.h5.lock')    
    return file_path, lock_path

def get_file_number(index):    
    file_nb = int(int(index) / get_file_size())     
    return file_nb

def get_file_numbers(start_index, stop_index):  
    start, stop = get_file_number(start_index),get_file_number(stop_index)     

    return [start, stop]

def exist(set_name ,index):        
    logger = logging.getLogger()

    file_path, lock_path = get_path(set_name ,index)
    # Locks the file since it will be accessed from multiple threads
    lock = FileLock(lock_path)
    lock.acquire()  
    file = None
    try:      
        file = h5py.File(file_path , "r")
        if (str(index) in file):
            return True
    except OSError as e:
        logger.info(set_name+"-- URL#"+str(index)+" Error: "+str(e))
    finally:
        if file != None:
            file.close()
        lock.release()
    return False

def store_status(set_name, index, status_code):
    logger = logging.getLogger()
    logger.info(set_name+"-- URL#"+str(index)+" Storing status--- "+str(status_code))
    
    file_path, lock_path = get_path(set_name ,index)
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
        logger.info(set_name+"-- URL#"+str(index)+" Error: "+str(e))
    finally:
        if file != None:
            file.close()
        lock.release()


def store_image(set_name, index, image, caption):
    """ Stores a single image/caption under the index in a given set_name storage
        Parameters:
        ---------------
        set_name    validation or training
        index       integer unique ID for image 
        image       image to be stored   
        caption     image caption
        
    """
    logger = logging.getLogger()
    logger.info(set_name+"-- URL#"+str(index)+" Storing image--- ")
    file_path, lock_path = get_path(set_name ,index)
    # Locks the file since it will be accessed from multiple threads

    lock = FileLock(lock_path)
    lock.acquire()  
    file = None    
    try:        
        # Create/retrieve a new HDF5 file
        file = h5py.File(file_path , "a")
               
        # Remove index from file if already exists
        if (str(index) in file):
            del file[str(index)]

        # Create a group/dataset in the file for a given image/caption
        group = file.create_group(str(index))
        group.create_dataset(
            "image", np.shape(image), h5py.h5t.STD_U8BE, data=np.asarray(image)
        )
        group.create_dataset(
            "caption", data=caption
        )
        group.create_dataset(
            "status", data="200"
        )
    except (OSError, Exception) as e:
        logger.info(set_name+"-- URL#"+str(index)+" Error: "+str(e))
    finally:
        if file != None:
            file.close()
        lock.release()



def read_image(set_name, index):
    """ Reads a single Image/Caption given its index out of a set_name storage
        Parameters:
        ---------------
        set_name    validation or training 
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
    file = h5py.File(file_path, "r")

    group = file[str(index)]

    status = np.array(group["status"]).astype("int32")
    if (status == 200):
        image = np.array(group["image"]).astype("uint8")
        caption = np.array(group["caption"]).astype("str")
        return status, image, caption
    
    return status, None, None


def get_last_stored_index(set_name):
    
    logger = logging.getLogger()

    f = []
    for (dirpath, dirnames, filenames) in walk('../data/img/'):
        f.extend(filenames)
        break
    
    filtered_list = list(filter(lambda i: (i.endswith(".h5") and i.find(set_name+"_") == 0), f))

    last_idx = str(len(filtered_list) -1)
    file = h5py.File(Path('../data/img/'+set_name+"_"+last_idx+".h5") , "a")
    keys_list = list(file.keys())
    logger.info("--KeysList size-- " + str(len(keys_list)))
    
    return len(keys_list) + get_file_size() * int(last_idx)

