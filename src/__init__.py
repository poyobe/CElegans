from skimage import io, morphology, measure, filter, draw
import numpy as np
import Image
import pylab
from pylab import *
import matplotlib.pyplot as plt
import math
import os

filePath = 'C:\\Users\\Dam\\Desktop\\ULB\\MA1\\Projet\\tests\\'

distBetweenCircles = 15
circleDiameter = 20   #il reste un bug quand je mets plus que 22 en circleDiameter
Cercles = True
Filter1 = False
Filter2 = True
CenterOfMass = True
BorderOfRegion = True

smallAreaSize = 4

thresholdVal = 5000

# test de seuil par Otsu trop eleve
"""
val2 = Filter.threshold_otsu(img)
print val
print val2
"""
"""------------------------------------------------------------------------------------------------------------"""
def measureFluo(x, y, circleDiameter, img):
    indice = 0
    rang = 1
    fluoSum = 0
    for i in range (indice, circleDiameter):
       fluoSum += img[x-circleDiameter/2+i][y]
       fluoSum += img[x-circleDiameter/2+i][y+rang]
       fluoSum += img[x-circleDiameter/2+i][y-rang]
    rang = 2
    for i in range (indice, circleDiameter):
       fluoSum += img[x-circleDiameter/2+i][y+rang]
       fluoSum += img[x-circleDiameter/2+i][y-rang]
    rang = 3
    for i in range (indice, circleDiameter):
       fluoSum += img[x-circleDiameter/2+i][y+rang]
       fluoSum += img[x-circleDiameter/2+i][y-rang]
     
    indice = 1
    rang = 4
    for i in range (indice, circleDiameter-indice):   
       fluoSum += img[x-circleDiameter/2+i][y-rang]
       fluoSum += img[x-circleDiameter/2+i][y+rang]
    rang = 5
    for i in range (indice, circleDiameter-indice):   
       fluoSum += img[x-circleDiameter/2+i][y-rang]
       fluoSum += img[x-circleDiameter/2+i][y+rang]
   
    indice = 2
    rang = 6
    for i in range (indice, circleDiameter-indice):   
       fluoSum += img[x-circleDiameter/2+i][y-rang]
       fluoSum += img[x-circleDiameter/2+i][y+rang]

    indice = 3
    rang = 7
    for i in range (indice, circleDiameter-indice):   
       fluoSum += img[x-circleDiameter/2+i][y-rang]
       fluoSum += img[x-circleDiameter/2+i][y+rang]
       
    indice = 4
    rang = 8
    for i in range (indice, circleDiameter-indice):   
       fluoSum += img[x-circleDiameter/2+i][y-rang]
       fluoSum += img[x-circleDiameter/2+i][y+rang]
       
    indice = 6
    rang = 9
    for i in range (indice, circleDiameter-indice):   
       fluoSum += img[x-circleDiameter/2+i][y-rang]
       fluoSum += img[x-circleDiameter/2+i][y+rang]
    
    fluoSum = fluoSum / 1000
    return fluoSum   

"""------------------------------------------------------------------------------------------------------------"""
def properties(fluo):
    
    numberOfPics = 0
    fluoSum = 0
    for i in range(0, len(fluo)):
        fluoSum += fluo[i]
        
    mean = fluoSum / len(fluo)
    
    picsThreshold = mean
    
    for i in range(0, len(fluo)):
        if fluo[i] > picsThreshold:
            numberOfPics += 1
    
    return (mean, picsThreshold, numberOfPics)
"""------------------------------------------------------------------------------------------------------------"""  
"""  
def initialisation(CerclesInit, Filter1Init, Filter2Init):
    Cercles = CerclesInit
    Filter1 = Filter1Init
    Filter2 = Filter2Init
    return(Cercles, Filter1, Filter2)   
"""  
"""------------------------------------------------------------------------------------------------------------"""    
def imgPreparation(img, thresholdVal):
    # Image en gris
    plt.subplot(2,2,1)
    plt.xlabel("image originale")
    io.imshow(img) # grayscale image
    
    #Seuillage + filtrage    
    plt.subplot(2,2,2)
    mask = img > thresholdVal
    if Filter1 == True:
        mask = morphology.remove_small_objects(mask)   #Attention le filtre ici n'est pas le meme que celui utilise apres la labelisation
        plt.xlabel("seuil :5000 + Filter1")
    else:
        plt.xlabel("seuil :5000")    
    io.imshow(mask)
    
    # Labelisation
    plt.subplot(2,2,3)
    if Filter1 == False and Filter2 == False:
        plt.xlabel("label")
    if Filter1 == False and Filter2 == True:
        plt.xlabel("label + Filter2")
    if Filter1 == True and Filter2 == False:
        plt.xlabel("label + Filter1")
    if Filter1 == True and Filter2 == True:
        plt.xlabel("label + Filter 1&2")
    label = morphology.label(mask, neighbors = 8, background=2)
    io.imshow(label, cmap='spectral')
    
    # Proprietes des regions lebelisees
    # Mesure des tailles des labels et de leur centre de masse
    props = measure.regionprops(label, ['Area','Centroid'])
    
    # Suppression des area de taille inferieure a smallAreaSize
    if Filter2 == True:
        j = len(props)
        i = 0
        while i < len(props):
            if props[i]['Area'] < smallAreaSize:
                props.remove(props[i])
                i = i-1
            i = i+1
            
    # Creation d'une liste des centres de masse
    centerofmass = []*len(props)
    for i in range(0,len(props)):
        centerofmass.append(props[i]['Centroid'])
        
    # relier les centres de masse des labels pour creer le neurone et l'afficher sur les labels
    x = []*len(centerofmass)
    y = []*len(centerofmass)

    for i in range(0,len(centerofmass)):
        x.append(centerofmass[i][1])
        y.append(centerofmass[i][0])

    plt.plot(x, y)
    
    # Construction du neurone avec ajout de points interpoles pour pouvoir mesurer la fluorescence 
    neuron = centerofmass

    # mesure de la distance entre 2 points du neurone consecutifs
    dist = []*(len(neuron)+1)
    for i in range(0, len(neuron)-1):
        dist.append(math.sqrt(((neuron[i][0]-neuron[i+1][0])*(neuron[i][0]-neuron[i+1][0]))+((neuron[i][1]-neuron[i+1][1])*(neuron[i][1]-neuron[i+1][1]))))

    # ajout des points intermediaire dans neuron
    distIndice = 0
    neuronIndice = 0
    for i in range(0,len(dist)):
        j = 1
        while dist[distIndice]/j > distBetweenCircles:  # determination de l'entier j qui divise la longueur entre 2 points en j morceaux
            j = j+1
        distIndice = distIndice + 1
        
        tx = linspace(neuron[neuronIndice][0], neuron[neuronIndice+1][0],j, endpoint=False)
        ty = linspace(neuron[neuronIndice][1], neuron[neuronIndice+1][1],j, endpoint=False)
    
        if len(tx) > 2:
            for z in range(1, len(tx)):
                neuron.insert(neuronIndice+1, (tx[z],ty[z]))
                neuronIndice = neuronIndice + 1
                
        neuronIndice += 1
        i += 1  
    
    # ajout des cercles sur le graphe si Cercles = True
    if Cercles == True:
        for i in range(0,len(neuron)):
            t = linspace(0, 2*pi, 20)
            x = circleDiameter/2*cos(t)
            y = circleDiameter/2*sin(t)
            plot(x+neuron[i][1], y+neuron[i][0])
            
    return neuron

"""------------------------------------------------------------------------------------------------------------""" 
def lenghtMeasure(neuron):
    dist2 = []*(len(neuron)-1)
    for i in range(0, len(neuron)-1):
        dist2.append(math.sqrt(((neuron[i][0]-neuron[i+1][0])*(neuron[i][0]-neuron[i+1][0]))+((neuron[i][1]-neuron[i+1][1])*(neuron[i][1]-neuron[i+1][1]))))

    distSum = []*(len(dist2))
    distSum.insert(0,0)

    for i in range(0, len(dist2)):
        distSum.append(distSum[i] + dist2[i])
    
    return distSum
    
"""-----------------------------------------------------------------------------------------------------------------"""  
def roundNumber(a):
    return floor(a+0.5)
"""-----------------------------------------------------------------------------------------------------------------"""  
def cElegans(filePath, Cercles, Filter1, Filter2, thresholdVal):
    listImg = os.listdir(filePath)

    imgPath = os.path.join(filePath, listImg[0])
    numberOfImg = len(listImg)
    for i in range(0,numberOfImg):
        plt.figure(i)
        imgPath = os.path.join(filePath, listImg[i])
        img = io.imread(imgPath)
        
        neuron = imgPreparation(img, thresholdVal)
        distSum = lenghtMeasure(neuron)
        fluo = []*(len(neuron)+1)
        for j in range(0,len(neuron)):
            fluo.append(measureFluo(neuron[j][0] , neuron[j][1], circleDiameter, img))
        
        propsGraph = properties(fluo)
        
    # graphe fluo + seuil des pics
        plt.subplot(2,2,4)
        # ajout du seuil des pics sur le graphe
        line = []*(len(distSum))
        round = roundNumber(propsGraph[1])
        for z in range(0,len(distSum)):
            line.append(round)
        plot(distSum, line)
        
        # graphe de la fluo
        plot(distSum, fluo)
        meanStr = str(propsGraph[0])
        picsThresholdStr = str(propsGraph[1])
        numberOfPicsStr = str(propsGraph[2])
        xlab = 'mean =' + meanStr +' / '+ 'picsThreshold =' + picsThresholdStr +' / '+ 'numberOfPics =' + numberOfPicsStr
        plt.xlabel(xlab)
        
        # sauvegarde des images
        pylab.savefig(listImg[i])
        
"""------------------------------------------------------------------------------------------------------------"""   
cElegans(filePath, Cercles, Filter1, Filter2, thresholdVal)

plt.show()

