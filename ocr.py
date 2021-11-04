# Trabalho de Processamento de Imagens (OCR)
# Curso: Ciência da Computação - 2021/2 - PUCMG
# Professor(a): Alexei Manso Corrêa Machado
# Alunos: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas e Umberto Castanheira
# Data da última modificação: 04/11/2021
# Arquivo: ocr.py 

from tkinter import *
from PIL import Image
from utils import *
from svm import *
from rede_profunda import *

# Baseline MLP for MNIST dataset
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
class Interface(object):
    def __init__(self):
        self.root = Tk()
        self.root.title = "Implementação de OCR"
        self.canvas = Canvas(self.root, bg='white', width=800, heigh=600)
        self.canvas.grid(row=3, columnspan=4)

        # Menu opções
        self.menuBar = Menu(self.root)

        self.options = Menu(self.menuBar, tearoff=0)
        self.options.add_command(label="Abrir imagem", command=self.open_image)
        self.menuBar.add_cascade(label="Opções", menu=self.options)
        self.root.config(menu=self.menuBar)

        # Menu funções
        self.options = Menu(self.menuBar, tearoff=0)
        self.options.add_command(label="Corrigir inclinação")
        self.options.add_command(label="Binarizar")
        self.options.add_command(label="Remoção de ruídos")
        self.options.add_command(label="Esqueleto da imagem")
        self.options.add_command(label="Escala")
        self.options.add_command(label="Treinar classificador Mahalanobis")
        self.options.add_command(label="Treinar SVM", command = trainSVM)
        self.options.add_command(label="Treinar rede neural", command = trainDeepLearning)
        self.options.add_command(label="Calcular matriz de confusão")
        self.menuBar.add_cascade(label="Funções", menu=self.options)
        self.root.config(menu=self.menuBar)

        # Menu sobre
        self.menuBar.add_command(label="Sobre", command=self.show_group)

        self.root.mainloop()

    def show_group(self):
        # Mostra na tela uma caixa de mensagem contendo o nome do grupo
        tkinter.messagebox.showinfo('Informações do trabalho', 'Processamento de Imagens - 2/2021 \nProfessor Alexei Machado \nIntegrantes: Ana Flávia Dias, Eduardo Pereira, Jonathan Douglas, Umberto Castanheira')

    # Função para abrir uma imagem selecionada
    def open_image(self):
        path = dlg.askopenfilename()
        if path != "":
            self.root.path = path
            load = Image.open(path)
            self.root.image_pillow = load
            self.root.image = image = ImageTk.PhotoImage(load)
            iw = image.width()
            ih = image.height()
            self.canvas.config(width=iw, height=ih)
            self.canvas.create_image(0, 0, image=image, anchor=NW)

if __name__ == '__main__':
    Interface()
