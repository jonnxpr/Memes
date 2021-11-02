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
import tensorflow as tf
import matplotlib.pyplot as plt

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

mnist = tf.keras.datasets.mnist #data base containing hand-written digits 0-9
(x_train,y_train), (x_test,y_test) = mnist.load_data()
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

 # normalizing the data to make the neural ntwork easier to learn
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#choosing the sequential model
model = tf.keras.models.Sequential()

#defining the architecture of the model
model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#defining the parameters to train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#training the model
model.fit(x_train,y_train,epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#saving the model
model.save("num_reader.model")
#load the model
new_model=tf.keras.models.load_model("num_reader.model")

predictions = new_model.predict([x_test])
print(predictions)
#prediction for first element in x_test is
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show() 
            
    

    
    
