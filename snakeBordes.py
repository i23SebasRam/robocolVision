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
    if dim[0] % divFilas != 0 or dim[1] % divColum != 0:
        fil = int((dim[0]/divFilas)) * divFilas
        col = int((dim[1]/divColum)) * divColum
        img = cv.resize(img,(col,fil))
        dim = img.shape
    
    pixHMin,pixWMin = dim[0]/divFilas,dim[1]/divColum
    matrizPosi = np.zeros((divFilas,divColum,2))

    for i in range(divFilas):
        for j in range(divColum):
            matrizPosi[i,j,:] = np.array([i*pixHMin,j*pixWMin])
    cosa = reorganizoMatriz(matrizPosi)
    

    

    return cosa
    
def reorganizoMatriz(matrizPosi):
    dim = matrizPosi.shape
    matrizPosi2 = np.zeros((dim[0],dim[1],8))
    for i in range(dim[0]-1):
        for j in range(dim[1]-1):
            arribaIzquierda = matrizPosi[i,j,:]
            arribaDerecha = matrizPosi[i,j+1,:]
            abajoDerecha = matrizPosi[i+1,j+1,:]
            abajoIzquierda = matrizPosi[i+1,j,:]
            matrizPosi2[i,j,:] = np.concatenate([arribaIzquierda,arribaDerecha,abajoDerecha,abajoIzquierda])
    return matrizPosi2

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

cosa = master(img1)
print(cosa)
