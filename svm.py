# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Data da última modificação: 04/11/2021
# Arquivo: svm.py

#Importações
import numpy as np
from utils import *

import sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Baseline MLP for MNIST dataset
from keras.datasets import mnist


def trainSVM():
    data = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = data
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    #normaliza os vetores
    X_train /= 255
    X_test /= 255
    
    for i in range(0,50):
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
        print("terminei")




