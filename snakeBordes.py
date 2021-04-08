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

prom = revisoParecido(img1,img2,mode=1)
print(prom)