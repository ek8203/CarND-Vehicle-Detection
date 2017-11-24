# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:48:46 2017

@author: nmkekrop
"""

import numpy as np
import cv2
import glob

def data_load(car_dir=None, notcar_dir=None, ext='png'):
    """
    Loads the vehicles and non-vehicle images from directories
    car_dir and notcar_dir
    """
    cars = []
    notcars = []
    for filename in glob.glob(car_dir + '/*/*.' + ext):
        #image = cv2.imread(filename)
        cars.append(filename)
    for filename in glob.glob(notcar_dir + '/*/*.' + ext):
        #image = cv2.imread(filename)
        notcars.append(filename) 
    return cars, notcars
