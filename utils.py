# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Data da última modificação: 04/11/2021
# Arquivo: utils.py

# Importações
import numpy as np
import cv2

import matplotlib.pyplot as plt


# requisito 2 - voltar aqui depois se necessário
def fixInclination(path):
    # carrega a imagem
    image = cv2.imread(path)
    #print(image)

    # corrige a escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    '''
    limiariza a imagem setando os pixels do primeiro plano
    para 255 e os do fundo para 0
    '''
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # captura as coordenadas dos pixels que são maiores do que 0
    # e computa a área de rotação que contém todas as coordenadas
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    #print("angulo da imagem = ", angle)

    if angle < -45:
        angle = -(90 + angle)
        #print("if")
    else:
        angle = -angle
        #print("else")

    if angle < -45:
        angle = 90 - abs(angle)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #print("[INFO] angle: {:.3f}".format(angle))
    #cv2.imshow("Input", image)
    #cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    
    return np.array(np.mat(rotated[0]))

# requisito 1 - pronto
def binarizeImage(im_gray, quant):
    #im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, quant, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite("bwimage.png", im_bw)
    return im_bw

def removeNoise(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    cv2.imshow('binary', out_binary)
    cv2.imwrite('binary.png', out_binary)

    cv2.imshow('gray', out_gray)
    cv2.imwrite('gray.png', out_gray)


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


def horizontalProjection(img1):

    #img1 = cv2.imread(image, 0)
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
    #img1 = cv2.imread(image, 0)
    ret, img1 = cv2.threshold(img1, 80, 255, cv2.THRESH_BINARY)

    (h, w) = img1.shape

    a = [0 for z in range(0, w)]

    for i in range(0, w):
        for j in range(0, h):
            if img1[j, i] == 0:
                a[i] += 1
                img1[j, i] = 255
    for i in range(0, w):
        for j in range(h-a[i], h):
            img1[j, i] = 0

    #cv2.imshow("img", img1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img1;

def skeletonize(path):
    # Read the image as a grayscale image
    img = cv2.imread(path, 0)
    
    # Threshold the image
    ret, img = cv2.threshold(img, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break

    # Displaying the final skeleton
    cv2.imshow("Skeleton", skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale(path, scaleProportion):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ', img.shape)
    scale_percent = scaleProportion # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)
    #cv2.imshow("Resized image", resized)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return resized

def getData(rangeFigure, baseX, baseY):
    fig = plt.figure()
    for i in range(rangeFigure):
        #print("comecei")
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(baseX[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(baseY[i]))
        plt.xticks([])
        plt.yticks([])
    fig.show()

def pixelDistribution(index, baseX, baseY):
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(baseX[index], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(baseY[index]))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,1,2)
    plt.hist(baseX[index].reshape(784))
    plt.title("Pixel Value Distribution")
    fig.show()


