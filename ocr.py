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
from utils import *

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
f = gzip.open('assets/img/baseMinst/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 5

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

i = 0
while (i < num_images):
    image = np.asarray(data[i]).squeeze()
    plt.imshow(image)
    plt.show()
    #image = fixInclination(image)
    #image = binarizeImage(image)
    
    i = i + 1

    
    
