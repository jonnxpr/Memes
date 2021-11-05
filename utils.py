# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Data da última modificação: 04/11/2021
# Arquivo: utils.py

import cv2

def horizontalProjection(img1):
    ret, img1 = cv2.threshold(img1, 80, 255, cv2.THRESH_BINARY) 

    (h, w) = img1.shape 

    a = [0 for z in range(0, h)] 

    for i in range(0, h): 
        for j in range(0, w): 
            if img1[i, j] == 0: 
                a[i] += 1 
                img1[i, j] = 255 
    for i in range(0, h): 
        for j in range(0, a[i]): 
            img1[i, j] = 0 
    #cv2.imshow("img", img1) 
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 
    
    return img1; 

def verticalProjection(img1):
    #image = cv2.imread(self.path)
    print(img1)
    ret, img1 = cv2.threshold(img1, 80, 255, cv2.THRESH_BINARY)

    (h, w) = img1

    a = [0 for z in range(0, w)]

    for i in range(0, w):
        for j in range(0, h):
            if img1[j, i] == 0:
                a[i] += 1
                img1[j, i] = 255
    for i in range(0, w):
        for j in range(h-a[i], h):
            img1[j, i] = 0

    cv2.imshow("img", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img1;