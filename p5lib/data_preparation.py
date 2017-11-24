# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:48:46 2017

@author: nmkekrop
"""

import numpy as np
import cv2
import glob
import matplotlib.image as mpimg

def data_load(car_dir=None, notcar_dir=None, ext='png'):
    """
    Load files wit the vehicles and non-vehicle images from directories
    car_dir and notcar_dir
    """
    cars = []
    notcars = []
    for filename in glob.glob(car_dir + '/*/*.' + ext):
        cars.append(filename)
    for filename in glob.glob(notcar_dir + '/*/*.' + ext):
        notcars.append(filename) 
    return cars, notcars

def data_look(car_list, notcar_list):
    """
    Get some characteristics of the dataset
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    image = mpimg.imread(car_list[10])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = image.shape    
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = image.dtype  
    # Return data_dict
    return data_dict
