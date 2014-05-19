'''
Created on 16 mai 2014

@author: Dam
'''
import Tkinter
from skimage import io, morphology, measure, filter, draw, exposure
import numpy as np
import Image
import pylab
from pylab import *
import matplotlib.pyplot as plt
import math
import os
from xlwt import Workbook
import ImageEnhance

filePath = 'C://Users/Dam/Desktop/ULB/MA1/Projet/tests'
"""filePath = 'C:\\Users\Dam\Desktop\ULB\MA1\Projet\tests'"""

 # ouverture du dossier et creation de la liste des images a scanner
listImg = os.listdir(filePath)
imgPath = os.path.join(filePath, listImg[0])
numberOfImg = len(listImg)

for i in range (0,numberOfImg):
    plt.figure(listImg[i])
    imgPath = os.path.join(filePath, listImg[i])
    img = io.imread(imgPath)
    subplot(1,2,1)
    io.imshow(img)
    """plt.show()"""
    subplot(1,2,2)
    """imgContrast = exposure.equalize_hist(img)"""
    imgContrast = exposure.equalize_adapthist(img, ntiles_x=8, ntiles_y=8, clip_limit=0.05)
    """imgContrast = exposure.equalize_adapthist(img)"""
    """imgContrast = exposure.rescale_intensity(img, out_range=(0, 65000))"""
    """
    amelioration = ImageEnhance.Contrast(img)
    imgContrast = amelioration.enhance(2.5)
    """
    
    io.imshow(imgContrast)
    
    plt.show()

    