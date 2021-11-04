# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Data da última modificação: 04/11/2021
# Arquivo: rede_profunda.py

import matplotlib.pyplot as plt
import os
from utils import *
from keras import *

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
    
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)
    
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
    
    history = model.fit(X_train, Y_train,epochs=20,verbose=2,validation_data=(X_test, Y_test))
    
    save_dir = "./results/"
    model_name = 'keras_mnist.h5'
    model_path = os.path.join(save_dir, model_name)
    print("path modelo = ", model_path)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()
    fig.show()