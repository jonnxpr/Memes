# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Versão: 1.0
# Data da última modificação: 27/10/2021
# Arquivo: ocr.py 

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import gzip

from numpy.lib.ufunclike import fix
from utils import *
import idx2numpy
from keras.datasets import mnist
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report

'''
    Processo:
    Pegar a imagem
    Corrigir a inclinação
    Binarizar
    Remover os ruídos
    Criar o esqueleto da imagem
'''

'''
    I had a project to detect license plates and these were the steps I did, 
    you can apply them to your project. After greying the image try applying 
    equalize histogram to the image, this allows the area's in the image with 
    lower contrast to gain a higher contrast. Then blur the image to reduce 
    the noise in the background. Next apply edge detection on the image, 
    make sure that noise is sufficiently removed as ED is susceptible to it. 
    Lastly, apply closing(dilation then erosion) on the image to close all the 
    small holes inside the words.
'''
data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data

arrayFinal = np.array([])

for i in range(0,50):
    array = np.array(np.mat(data[0][0][i]))
    projHor = horizontalProjection(array)
    projVert = verticalProjection(array)
    projConcat = projHor + projVert
    clf = sklearn.svm.SVC(gamma=0.001, C=100)
    clf.fit(projHor, np.array(projVert[0], ndmin = 1))
    #predicted = clf.predict(np.array(X_test[0], ndmin = 2))
    #plt.imshow(predicted)
    #plt.show()
    #print(
    #f"Classification report for classifier {clf}:\n"
    #f"{sklearn.metrics.classification_report(np.array(y_test[0], ndmin=1), predicted)}\n")
    #plt.imshow(projConcat)
    #plt.show()
    #array = binarizeImage(array)
    #array = fixInclination("bwimage.png")
    #print(array.shape)
    #print(array)

predicted = clf.predict(np.array(X_test[0], ndmin = 1))
print(predicted)

    
    
