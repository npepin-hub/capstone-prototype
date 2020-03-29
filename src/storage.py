import pandas as pd
import requests
from PIL import Image
import io
import os
from pathlib import Path
import h5py
import numpy as np

def exist(set_name ,index):
    file = h5py.File(Path('../data/img/'+set_name+'.h5') , "a")
    if (str(index) in file):
        return True

    return False

def store_status(set_name, index, status_code):
    # Create/retrieve a new HDF5 file
    file = h5py.File(Path('../data/img/'+set_name+'.h5') , "a")
    
    # Remove image_id from file if already exists
    if (str(index) in file):
        del file[str(index)]

    # Create a group/dataset in the file for a given image/caption
    group = file.create_group(str(index))
    meta_set = group.create_dataset(
        "status", data=status_code
    )
    
    file.close()    

    return


def store_image(set_name, index, image, caption):
    """ Stores a single image and its caption.
        Parameters:
        ---------------
        image       image to be stored
        index       integer unique ID for image
        caption     image caption
        set_name    validation or training 
    """
    # Create/retrieve a new HDF5 file
    file = h5py.File(Path('../data/img/'+set_name+'.h5') , "a")
    
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
    
    file.close()    


def read_image(set_name, index):
    """ Reads a single Image/Caption out of a storage
        Parameters:
        ---------------
        image_id    integer unique ID for image
        set_name    validation or training 

        Returns:
        ----------
        image       image array stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(Path('../data/img/'+set_name+'.h5'), "r+")

    group = file[str(index)]


    status = np.array(group["status"]).astype("str")
    print(type(status))
    if (int(status) == 200):
        image = np.array(group["image"]).astype("uint8")
        caption = np.array(group["caption"]).astype("str")
        return status, image, caption
    
    return status, None, None
 

 

