import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


#Imagen 1, toda la imagen es igual (Verde).
img1 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen7.jpg')
#Imagen 2, tiene formas interesantes para probar como funciona el algoritmo.
img2 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen8.jpg')
#Imagen 3, es una imagen modelo de lo que veria el robert.
img3 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/cosita2.jpg')

def master(img,divFilas = 10,divColum = 10):
    dim = img.shape
    #Se hace un escalamiento para que la imagen se pueda dividir en partes iguales.
    if dim[0] % divFilas != 0:
        fil = int(dim[0]/divFilas) * divFilas
        img = cv.resize(img,(fil,dim[1]))
    elif dim[1] % divColum != 0:
        col = int(dim[1]/divColum) * divColum
        img = cv.resize(img,(dim[0],col))
    


#Funcion que recibe los indices y entrega la imagen pequeñita.
#ind, un vector fila donde el primer valor es la esquina izquierda abajo y va en sentido horario del cuadrado.
def pasameIndices(imagen,ind):
    return imagen[ind[1]:ind[0],ind[2]:ind[3]]
    


#Vamos a hacer una funcion que revise la distancia euclidiana de los tres canales.
def revisoParecido(seccionActual,seccionComparar,mode = 0):
    dis = 0
    promActu = cv.mean(seccionActual)
    promComp = cv.mean(seccionComparar)
    #Distancia euclidiana.
    if mode == 0:
        disEucli = math.sqrt((promActu[0]- promComp[0])**2.0 + (promActu[1]- promComp[1])**2.0 +(promActu[2]- promComp[2])**2.0 )
        dis = disEucli
    #RMSE penaliza distancias lejanas
    if mode == 1:
        RMSE = math.sqrt(((promActu[0]- promComp[0])**2 + (promActu[1]- promComp[1])**2 +(promActu[2]- promComp[2])**2)/3 )
        dis = RMSE
    #Distancia manhattan.
    if mode == 2:
        disMan = abs(promActu[0]- promComp[0]) + abs(promActu[1]- promComp[1]) + abs(promActu[2]- promComp[2]) 
        dis = disMan
    return dis

imgen = pasameIndices(img1,np.array([630,610,300,630]))
cv.imshow("imagen",imgen)

