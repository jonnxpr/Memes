# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Versão: 1.0
# Data da última modificação: 27/10/2021
# Arquivo: utils.py 

#Importações
import numpy as np
import cv2
from PIL import Image

def openImage(path):
    #carrega uma imagem e mostra
    im = Image.open(path)
    im.show()

def fixInclination(path):
    #carrega a imagem
    image = cv2.imread(path)

    #corrige a escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    '''
    limiariza a imagem setando os pixels do primeiro plano
    para 255 e os do fundo para 0
    '''
    thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #captura as coordenadas dos pixels que são maiores do que 0
    #e computa a área de rotação que contém todas as coordenadas
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Input", image)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

def binarizeImage(path):
    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("bwimage.png", im_bw)
    
def removeNoise(path):
    image = cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(image, bg, scale=255)
    out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

    cv2.imshow('binary', out_binary)  
    cv2.imwrite('binary.png',out_binary)

    cv2.imshow('gray', out_gray)  
    cv2.imwrite('gray.png',out_gray)

def equalizeHistogram(path):
    src = cv2.imread(cv2.samples.findFile(path))
    if src is None:
        print('Could not open or find the image:', path)
        exit(0)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(src)
    cv2.imshow('Source image', src)
    cv2.imshow('Equalized Image', dst)
    cv2.waitKey()

equalizeHistogram("assets/img/lenna.png")

