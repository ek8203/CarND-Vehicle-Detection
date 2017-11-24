# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:23:52 2017

@author: nmkekrop
"""

import matplotlib.pyplot as plt

"""
Plot one image with title
"""
def plt_one(image, title = None, cmap=None, figsize=None, fontsize=None):
    figure, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=fontsize)
    plt.show()
    return figure

"""
Plot 2 images with titles
"""    
def plt_two(image_1, image_2, title_1 = None, title_2 = None, cmap_1=None, 
            cmap_2=None, figsize=None, fontsize=None):
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image_1, cmap=cmap_1)
    ax1.set_title(title_1, fontsize=fontsize)
    ax2.imshow(image_2, cmap=cmap_2)
    ax2.set_title(title_2, fontsize=fontsize)
    plt.show()
    return figure