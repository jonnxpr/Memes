# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Versão: 1.0
# Data da última modificação: 25/10/2021
# Arquivo: svm.py

#Importações
import svm
from scipy.spatial import distance
import scipy
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import gzip

from numpy.lib.ufunclike import fix
from utils import *
from keras.datasets import mnist
import keras
from keras import *
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Baseline MLP for MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import np_utils

def trainDeepLearning():
    data = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = data
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    #normaliza os vetores
    X_train /= 255
    X_test /= 255
    
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    
    
    '''for i in range(0,50):
        array = np.array(np.mat(data[0][0][i]))
        projHor = horizontalProjection(array)
        projVert = verticalProjection(array)
        projConcat = projHor + projVert
        clf = sklearn.svm.SVC(gamma='auto', C=100)
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

        predicted = clf.predict(np.array(np.mat(data[0][0][i])))
        print(predicted)
        print ("accuracy",sklearn.metrics.accuracy_score(np.array(y_train, ndmin = 1), predicted))
        print("terminei")'''




