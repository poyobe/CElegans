'''
Created on 20 avr. 2014

@author: Dam
'''
import Tkinter
from skimage import io, morphology, measure, filter, draw
import numpy as np
import Image
import pylab
from pylab import *
import matplotlib.pyplot as plt
import math
import os
from xlwt import Workbook

smallAreaSize = 4

# test de seuil par Otsu (trop eleve)
"""
val2 = Filter.threshold_otsu(img)
print val
print val2
"""
"""------------------------------------------------------------------------------------------------------------"""
# Mesure de la fluoNeuron dans un cercle de rayon circleDiameter/2 centre en (x,y)
def measurefluoNeuron(x, y, circleDiameter, img):
    # Mesure de la fluoNeuron dans un cercle de rayon circleDiameter/2 centre en (x,y)
    fluoNeuronSum = 0
    for i in range (0, circleDiameter):
        for j in range (0, circleDiameter):
            
            posX = np.asscalar(x - circleDiameter/2 + i)
            posY = np.asscalar(y - circleDiameter/2 + j)
            dist = math.sqrt((posX-x)*(posX-x)+(posY-y)*(posY-y))

            if dist < circleDiameter/2 and 0<posX<512 and 0<posY<512:
                fluoNeuronSum += img[posX][posY]
    
    fluoNeuronSum = fluoNeuronSum / 1000
    return fluoNeuronSum  
"""------------------------------------------------------------------------------------------------------------"""

# Mesure des proprietes liees a la fluoNeuron de l'image
def properties(fluoNeuron, picsThreshold, fluoMax, fluoMaxIndice):
    
    numberOfPicsThreshold = 0
    numberOfPics = 0
    fluoNeuronSum = 0
    fluoNeuronSumWOSoma = 0
    
    # mesure de la moyenne de fluoNeuron sur tout le neurone
    for i in range(0, len(fluoNeuron)):
        fluoNeuronSum += fluoNeuron[i]
        
    mean = fluoNeuronSum / len(fluoNeuron)
    
    # mesure de la moyenne de fluoNeuron sur tout le neurone sauf le noyau
    if fluoMaxIndice == 0 or fluoMaxIndice == 1: # on tient compte du cas ou le noyau est au bout du neurone
        for i in range(fluoMaxIndice+2, len(fluoNeuron)):
            fluoNeuronSumWOSoma += fluoNeuron[i]
    if fluoMaxIndice == len(fluoNeuron) or fluoMaxIndice == len(fluoNeuron)-1: # on tient compte du cas ou le noyau est au bout du neurone
        for i in range(0, fluoMaxIndice-2):
            fluoNeuronSumWOSoma += fluoNeuron[i]
    
    if 1 < fluoMaxIndice < len(fluoNeuron)-1: # on somme les valeurs de fluoNeuron sans prendre le noyau
        for i in range(0, fluoMaxIndice-2):
            fluoNeuronSumWOSoma += fluoNeuron[i]
        for i in range(fluoMaxIndice+2, len(fluoNeuron)):
            fluoNeuronSumWOSoma += fluoNeuron[i]
                
    # mesure de la moyenne dans le Soma en tenant compte de sa position dans le neurone
    if fluoMaxIndice == 0:
        meanSoma = (fluoNeuron[0] + fluoNeuron[1])/2
    if fluoMaxIndice == len(fluoNeuron)-1:
        meanSoma = (fluoNeuron[fluoMaxIndice-1] + fluoNeuron[fluoMaxIndice])/2
    else:
        meanSoma = (fluoNeuron[fluoMaxIndice-1] + fluoNeuron[fluoMaxIndice] + fluoNeuron[fluoMaxIndice+1])/3
    
    meanWOSoma = fluoNeuronSumWOSoma / len(fluoNeuron)
    
    # compte du nombre de pics au dessus du seuil picsThreshold et du nombre de pics absolus
    above = False
    
    if fluoNeuron[0] > picsThreshold:
        above = True
        numberOfPicsThreshold += 1
          
    for i in range(0, len(fluoNeuron)):
        if fluoNeuron[i] > picsThreshold and above == False:
            numberOfPicsThreshold += 1
            above = True
        if fluoNeuron[i] <= picsThreshold:
            above = False
            
    # nombre de pics absolus
    for i in range(2, len(fluoNeuron)-3):        
        if fluoNeuron[i-1]<fluoNeuron[i]>fluoNeuron[i+1] and fluoNeuron[i-2]<fluoNeuron[i-1] and fluoNeuron[i+2]<fluoNeuron[i+1]:
            numberOfPics += 1
            
    # nombre de pics absolus - cas particuliers des bords      
    if fluoNeuron[0] > fluoNeuron[1] > fluoNeuron[2]:
        numberOfPics += 1
    if fluoNeuron[len(fluoNeuron)-3] < fluoNeuron[len(fluoNeuron)-2] < fluoNeuron[len(fluoNeuron)-1]:
        numberOfPics += 1
                           
    return (mean, picsThreshold, numberOfPicsThreshold, numberOfPics, meanWOSoma, meanSoma)
"""------------------------------------------------------------------------------------------------------------"""  

"""------------------------------------------------------------------------------------------------------------"""    
def imgPreparation(img, Cercles, Filter1, Filter2, distBetweenCircles, circleDiameter, thresholdVal):
    # Image en gris
    plt.subplot(2,2,1)
    plt.xlabel("original image")
    io.imshow(img) # grayscale image
    
    #Seuillage + filtrage    
    plt.subplot(2,2,2)
    mask = img > thresholdVal
    
    if Filter1 == True:
        mask = morphology.remove_small_objects(mask, min_size=7)   #Attention le filtre ici n'est pas le meme que celui utilise apres la labelisation
        plt.xlabel("seuil :" +str(thresholdVal) + "+ Filter1")
    else:
        plt.xlabel("seuil :" +str(thresholdVal))    
    io.imshow(mask)
    
    # Labelisation
    plt.subplot(2,2,3)
    if Filter1 == False and Filter2 == False:
        plt.xlabel("labels")
    if Filter1 == False and Filter2 == True:
        plt.xlabel("labels + Filter2")
    if Filter1 == True and Filter2 == False:
        plt.xlabel("labels + Filter1")
    if Filter1 == True and Filter2 == True:
        plt.xlabel("labels + Filter 1&2")
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
    
    # Construction du neurone avec ajout de points interpoles pour pouvoir mesurer la fluoNeuronrescence 
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
            
        while dist[distIndice]/j > float(distBetweenCircles):  # determination de l'entier j qui divise la longueur entre 2 points en j morceaux
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
def lenghtMeasure(neuron, start, stop):
    dist = []*((stop-start))
    for i in range(start, stop-1):
        dist.append(math.sqrt(((neuron[i][0]-neuron[i+1][0])*(neuron[i][0]-neuron[i+1][0]))+((neuron[i][1]-neuron[i+1][1])*(neuron[i][1]-neuron[i+1][1]))))
    
    # gestion du cas particulier ou la longueur du dendrite est nulle
    distSum = []*(len(dist))
    distSum.append(0)
    if len(dist)>0:
        distSum.append(dist[0])
    if len(dist)>1:
        for i in range(2, len(dist)):
            distSum.append(distSum[i-1] + dist[i-1])
        distSum.append(distSum[-1] + dist[-1])
    
    return (distSum, distSum[-1])
    
"""-----------------------------------------------------------------------------------------------------------------"""  
def roundNumber(a):
    return floor(a+0.5)
"""-----------------------------------------------------------------------------------------------------------------"""  

def saveParameters(feuil1, listImg, i, propsGraph, distSum, fluoMax, distDendSum, distAxonSum):
    feuil1.write(i+5,0, listImg)
    feuil1.write(i+5,1,propsGraph[0])
    feuil1.write(i+5,2,propsGraph[4])
    feuil1.write(i+5,3,int(distSum))
    feuil1.write(i+5,4,fluoMax)
    feuil1.write(i+5,5,propsGraph[2])
    feuil1.write(i+5,6,propsGraph[3])
    feuil1.write(i+5,7,propsGraph[5])
    feuil1.write(i+5,8,int(distDendSum))
    feuil1.write(i+5,9,int(distAxonSum))
    
    
"""-----------------------------------------------------------------------------------------------------------------"""  
def cElegans(filePath, Cercles, Filter1, Filter2, distBetweenCircles, circleDiameter, thresholdVal, picsThreshold, circleDiameterSoma):
    
    # ouverture du dossier et creation de la liste des images a scanner
    listImg = os.listdir(filePath)
    imgPath = os.path.join(filePath, listImg[0])
    numberOfImg = len(listImg)
    
    # creation du workbook
    book = Workbook()
 
    # creation de la feuille 1
    feuil1 = book.add_sheet('feuille 1')
 
    # ajout des en-tetes et des parametres du scan
    feuil1.write(0,1,'Filter1')
    feuil1.write(0,2,'Filter2')
    feuil1.write(0,3,'distBetweenCircles')
    feuil1.write(0,4,'circleDiameter')
    feuil1.write(0,5,'circleDiameterSoma')
    feuil1.write(0,6,'picsThreshold')
    feuil1.write(0,7,'thresholdVal')
    
    feuil1.write(1,1, str(Filter1))
    feuil1.write(1,2, str(Filter2))
    feuil1.write(1,3, int(distBetweenCircles))
    feuil1.write(1,4, int(circleDiameter))
    feuil1.write(1,5, int(circleDiameterSoma))
    feuil1.write(1,6, int(picsThreshold))
    feuil1.write(1,7, int(thresholdVal))
    
    feuil1.write(3,0,'ID')
    feuil1.write(3,1,'mean')
    feuil1.write(3,2,'meanWithoutSoma')
    feuil1.write(3,3,'neuron lenght')
    feuil1.write(3,4,'max')
    feuil1.write(3,5,'numberOfPicsOverTreshold')
    feuil1.write(3,6,'numberOfPics')
    feuil1.write(3,7,'meanSoma')
    feuil1.write(3,8,'dendrite lenght')
    feuil1.write(3,9,'axon lenght')
    
    """
    # ajustement de la taille des colonnes dans le xls
    for j in range(0,10):
        feuil1.col(j).width = 6000
    """
    
    """
    # mesure du ratio aire des cercles de l'axone / aire des cercles du soma pour normaliser la mesure de fluo par rapport a la surface des cercles
    ratioArea = (math.pi*(circleDiameter/2)*(circleDiameter/2))/(math.pi*(circleDiameterSoma/2)*(circleDiameterSoma/2))
    """
     
    # boucle sur toutes les images pour appliquer le scan   
    for i in range(0,numberOfImg):
        plt.figure(listImg[i])
        imgPath = os.path.join(filePath, listImg[i])
        img = io.imread(imgPath)
        
        neuron = imgPreparation(img, Cercles, Filter1, Filter2, distBetweenCircles, circleDiameter, thresholdVal)
        distSum = lenghtMeasure(neuron, 0, len(neuron))
        fluoNeuron = []*(len(neuron)+1)
        
        fluoMax = 0
        fluoMaxIndice = 0
        for j in range(0,len(neuron)):
            fluoNeuron.append(measurefluoNeuron(neuron[j][0] , neuron[j][1], circleDiameter, img))
            
            # dtermination du max de fluo et de sa position dans le neurone
            if fluoNeuron[j] > fluoMax:
                fluoMax = fluoNeuron[j]
                fluoMaxIndice = j
        
        
        # mesure des longueurs de chaque cote du soma pour avoir la longueur du dendrite et de l'axone
        dist1 = lenghtMeasure(neuron, fluoMaxIndice, len(neuron))
        dist2 = lenghtMeasure(neuron, 0, fluoMaxIndice)
        
        distDendSum = min(int(dist1[1]), int(dist2[1]))
        distAxonSum = max(int(dist1[1]), int(dist2[1]))
        
        # determination du nombre de points ou aggrandir le cercle si le noyau est proche du bord
        # + re-mesure de la fluo dans un cercle plus grand pour le noyau
        if fluoMaxIndice == 0 or fluoMaxIndice == 1:
            rangeSoma = fluoMaxIndice + 2
            for r in range(0, rangeSoma):
                fluoNeuron[r] = measurefluoNeuron(neuron[r][0] , neuron[r][1], circleDiameterSoma, img)
                
            # ajout des cercles plus grands sur le graphe si Cercles = True
                if Cercles == True:
                    t = linspace(0, 2*pi, 20)
                    x = circleDiameterSoma/2*cos(t)
                    y = circleDiameterSoma/2*sin(t)
                    plot(x+neuron[r][1], y+neuron[r][0])
                    
        if fluoMaxIndice == len(neuron) or fluoMaxIndice == len(neuron)-1:
            rangeSoma = fluoMaxIndice - 2
            for r in range(rangeSoma, len(neuron)):
                fluoNeuron[r] = measurefluoNeuron(neuron[r][0] , neuron[r][1], circleDiameterSoma, img)
                
            # ajout des cercles plus grands sur le graphe si Cercles = True
                if Cercles == True:
                    t = linspace(0, 2*pi, 20)
                    x = circleDiameterSoma/2*cos(t)
                    y = circleDiameterSoma/2*sin(t)
                    plot(x+neuron[r][1], y+neuron[r][0])
                    
        if 1 < fluoMaxIndice < len(neuron)-1:                                     
            for k in range(fluoMaxIndice-2, fluoMaxIndice+2):
                fluoNeuron[k] = measurefluoNeuron(neuron[k][0] , neuron[k][1], circleDiameterSoma, img)
                
                # ajout des cercles plus grands sur le graphe si Cercles = True
                if Cercles == True:
                    t = linspace(0, 2*pi, 20)
                    x = circleDiameterSoma/2*cos(t)
                    y = circleDiameterSoma/2*sin(t)
                    plot(x+neuron[k][1], y+neuron[k][0])                                        
    
        propsGraph = properties(fluoNeuron, picsThreshold, fluoMax, fluoMaxIndice)
        
        # graphe fluoNeuron + seuil des pics
        plt.subplot(2,2,4)
        # ajout du seuil des pics sur le graphe
        line = []*(len(distSum[0]))
        round = roundNumber(propsGraph[1])
        for z in range(0,len(distSum[0])):
            line.append(round)
        plot(distSum[0], line)
        
        # graphe de la fluoNeuron
        plot(distSum[0], fluoNeuron)
        meanStr = str(propsGraph[0])
        picsThresholdStr = str(propsGraph[1])
        numberOfPicsStr = str(propsGraph[2])
        xlab = 'mean =' + meanStr +' / '+ 'picsThreshold =' + picsThresholdStr +' / '+ 'numberOfPics =' + numberOfPicsStr
        plt.xlabel(xlab)
        
        # sauvegarde des images et du tableur
        pylab.savefig(listImg[i])
        # creation materielle du fichier resultant
        saveParameters(feuil1, str(listImg[i]), i, propsGraph, distSum[1], fluoMax, distDendSum, distAxonSum)
        
    book.save('CElegans.xls')    
"""------------------------------------------------------------------------------------------------------------"""   

class simpleapp_tk(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()
        
    def initialize(self):
        self.grid()
        #ajout du champ pour copier le filePath et du bouton pour lancer le scan
        self.filePathVariable = Tkinter.StringVar()
        self.filePath = Tkinter.Entry(self, textvariable=self.filePathVariable)
        self.filePath.grid(column=0,row=0,sticky='WE')
        self.filePath.bind("<Return>", self.OnPressEnter)
        self.filePathVariable.set(u"C:\\Users\\Dam\\Desktop\\ULB\\MA1\\Projet\\tests\\")
        
        filePathButton = Tkinter.Button(self,text=u"Scan", command=self.OnButtonClick)
        filePathButton.grid(column=1,row=0)
        
        # ajout des champs pour les filtres, distBetweenCircles, circleDiameter, cercles
        self.FilterVariable = Tkinter.StringVar()
        self.Filter = Tkinter.Entry(self, textvariable=self.FilterVariable)
        self.Filter.grid(column=0,row=1,sticky='WE')
        self.FilterVariable.set(u"I")
        
        self.Filter2Variable = Tkinter.StringVar()
        self.Filter2 = Tkinter.Entry(self, textvariable=self.Filter2Variable)
        self.Filter2.grid(column=0,row=2,sticky='WE')
        self.Filter2Variable.set(u"A")
        
        self.distBetweenCirclesVariable = Tkinter.StringVar()
        self.distBetweenCircles = Tkinter.Entry(self, textvariable=self.distBetweenCirclesVariable)
        self.distBetweenCircles.grid(column=0,row=3,sticky='WE')
        self.distBetweenCirclesVariable.set(u"10")
        
        self.circleDiameterVariable = Tkinter.StringVar()
        self.circleDiameter = Tkinter.Entry(self, textvariable=self.circleDiameterVariable)
        self.circleDiameter.grid(column=0,row=4,sticky='WE')
        self.circleDiameterVariable.set(u"12")
        
        self.cerclesVariable = Tkinter.StringVar()
        self.cercles = Tkinter.Entry(self, textvariable=self.cerclesVariable)
        self.cercles.grid(column=0,row=5,sticky='WE')
        self.cerclesVariable.set(u"A")
        
        self.thresholdValVariable = Tkinter.StringVar()
        self.thresholdVal = Tkinter.Entry(self, textvariable=self.thresholdValVariable)
        self.thresholdVal.grid(column=0,row=6,sticky='WE')
        self.thresholdValVariable.set(u"5000")
        
        self.picsThresholdVariable = Tkinter.StringVar()
        self.picsThreshold = Tkinter.Entry(self, textvariable=self.picsThresholdVariable)
        self.picsThreshold.grid(column=0,row=7,sticky='WE')
        self.picsThresholdVariable.set(u"600")
        
        self.circleDiameterSomaVariable = Tkinter.StringVar()
        self.circleDiameterSoma = Tkinter.Entry(self, textvariable=self.circleDiameterSomaVariable)
        self.circleDiameterSoma.grid(column=0,row=8,sticky='WE')
        self.circleDiameterSomaVariable.set(u"22")
        
        # ajout des labels pour les filtres, distBetweenCircles, circleDiameter, cercles
        self.FilterLabelVariable = Tkinter.StringVar()
        FilterLabel = Tkinter.Label(self, textvariable=self.FilterLabelVariable, anchor="w",fg="white",bg="blue")
        FilterLabel.grid(column=1,row=1,columnspan=2,sticky='WE')
        self.FilterLabelVariable.set(u"Filter1 (A - Active / I - Inactive)")
        
        self.Filter2LabelVariable = Tkinter.StringVar()
        Filter2Label = Tkinter.Label(self, textvariable=self.Filter2LabelVariable, anchor="w",fg="white",bg="blue")
        Filter2Label.grid(column=1,row=2,columnspan=2,sticky='WE')
        self.Filter2LabelVariable.set(u"Filter2 (A - Active / I - Inactive)")
        
        self.distBetweenCirclesLabelVariable = Tkinter.StringVar()
        distBetweenCirclesLabel = Tkinter.Label(self, textvariable=self.distBetweenCirclesLabelVariable, anchor="w",fg="white",bg="blue")
        distBetweenCirclesLabel.grid(column=1,row=3,columnspan=2,sticky='WE')
        self.distBetweenCirclesLabelVariable.set(u"distBetweenCircles (3-16)")
        
        self.circleDiameterLabelVariable = Tkinter.StringVar()
        circleDiameterLabel = Tkinter.Label(self, textvariable=self.circleDiameterLabelVariable, anchor="w",fg="white",bg="blue")
        circleDiameterLabel.grid(column=1,row=4,columnspan=2,sticky='WE')
        self.circleDiameterLabelVariable.set(u"circleDiameter (0-22)")
        
        self.cerclesLabelVariable = Tkinter.StringVar()
        cerclesLabel = Tkinter.Label(self, textvariable=self.cerclesLabelVariable, anchor="w",fg="white",bg="blue")
        cerclesLabel.grid(column=1,row=5,columnspan=2,sticky='WE')
        self.cerclesLabelVariable.set(u"cercles (A - Active / I - Inactive)")
        
        self.thresholdValLabelVariable = Tkinter.StringVar()
        thresholdValLabel = Tkinter.Label(self, textvariable=self.thresholdValLabelVariable, anchor="w",fg="white",bg="blue")
        thresholdValLabel.grid(column=1,row=6,columnspan=2,sticky='WE')
        self.thresholdValLabelVariable.set(u"thresholdVal (4500 - 6500)")
        
        self.picsThresholdLabelVariable = Tkinter.StringVar()
        picsThresholdLabel = Tkinter.Label(self, textvariable=self.picsThresholdLabelVariable, anchor="w",fg="white",bg="blue")
        picsThresholdLabel.grid(column=1,row=7,columnspan=2,sticky='WE')
        self.picsThresholdLabelVariable.set(u"picsThreshold")
        
        self.circleDiameterSomaLabelVariable = Tkinter.StringVar()
        circleDiameterSomaLabel = Tkinter.Label(self, textvariable=self.circleDiameterSomaLabelVariable, anchor="w",fg="white",bg="blue")
        circleDiameterSomaLabel.grid(column=1,row=8,columnspan=2,sticky='WE')
        self.circleDiameterSomaLabelVariable.set(u"circleDiameterSoma (22)")
        
        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,True)
        
    def OnButtonClick(self):
        if self.cerclesVariable.get() == 'A':
            Cercles = True
        else:
            Cercles = False
        if self.FilterVariable.get() == 'A':
            Filter1 = True
        else:
            Filter1 = False
        if self.Filter2Variable.get() == 'A':
            Filter2 = True
        else:
            Filter2 = False
        
        cElegans(self.filePathVariable.get(), Cercles, bool(Filter1), bool(Filter2), self.distBetweenCirclesVariable.get(), int(self.circleDiameterVariable.get()), float(self.thresholdValVariable.get()), float(self.picsThresholdVariable.get()), int(self.circleDiameterSomaVariable.get()))
        plt.show()
        
    def OnPressEnter(self,event):
        if self.cerclesVariable.get() == 'A':
            Cercles = True
        else:
            Cercles = False
        if self.FilterVariable.get() == 'A':
            Filter1 = True
        else:
            Filter1 = False
        if self.Filter2Variable.get() == 'A':
            Filter2 = True
        else:
            Filter2 = False
        
        cElegans(self.filePathVariable.get(), Cercles, bool(Filter1), bool(Filter2), self.distBetweenCirclesVariable.get(), int(self.circleDiameterVariable.get()), float(self.thresholdValVariable.get()), float(self.picsThresholdVariable.get()), int(self.circleDiameterSomaVariable.get()))
        plt.show()
        
if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('C Elegans') 
    app.mainloop()