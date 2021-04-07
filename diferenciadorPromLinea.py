#Aca vamos a colocar el codigo final que realiza el promedio y obtiene una linea, en donde separa el cielo y la tierra.
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""Importamos las imagenes"""
img1 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen1.jpg')
img2 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen2.jpg')
img3 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen3.jpg')
img4 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen4.jpg')
img5 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen5.jpg')
img6 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen6.jpg')
img7 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen7.jpg')
img8 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen8.jpg')
img9 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/Imagen9.jpg')
img10 = cv.imread('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/cosita2.jpg')
video = cv.VideoCapture('C:/Users/pc/OneDrive - Universidad de los andes/Septimo semestre/Robocol/Img/RosbagFree.mp4')

#Imagen que se quiere analizar.
img = img2

#Obtencion de la imagen en otros canales.
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)

#Tamaño de la imagen
dimensionesImg = img.shape
W = dimensionesImg[1]
H = dimensionesImg[0]

#Procesamiento que se le hace a la imagen.
imgPrueba = cv.GaussianBlur(img,(5,5),0)
imgPrueba = cv.pyrMeanShiftFiltering(imgPrueba,sp=15,sr=40,maxLevel = 3)
imgHSV = cv.cvtColor(imgPrueba,cv.COLOR_BGR2HSV)
imgGray = cv.cvtColor(imgPrueba,cv.COLOR_BGR2GRAY)
bordes = cv.Canny(imgPrueba,250,400)
tamaño = np.asarray(bordes)

#Base del codigo, la cual realiza el promedio.
dame = np.array(np.where(tamaño == 255))
prom = int(np.mean(dame[0,:]))
cv.line(imgPrueba,(0,prom),(W,prom),(0,0,255))

#Cuantos pixeles va a tomar de la imagen.
pix = 5

#img[alto(H),ancho(W)]

imgCutUp = imgGray[0:pix,0:W]
imgCutMiddle = imgGray[int((H/2)-(pix/2)):int((H/2)+(pix/2)),0:W]
imgCutDown = imgGray[H-pix:H,0:W]

cv.line(imgPrueba,(0,pix),(W,pix),(0,255,0))
cv.line(imgPrueba,(0,H-pix),(W,H-pix),(0,255,0))


promImgCutup = np.mean(imgCutUp)
print(promImgCutup)
promImgCutMiddle = np.mean(imgCutMiddle)
print(promImgCutMiddle)
promImgCutDown = np.mean(imgCutDown)
print(promImgCutDown)


cv.imshow('imagenSuave',imgPrueba)

cv.imshow('bordes',bordes)

#Histogramas.
histH = cv.calcHist([imgPrueba], [0], None, [180], [0, 180])
histH = histH/max(histH)
histS = cv.calcHist([imgPrueba], [1], None, [256], [0, 256])
histS = histS/max(histS)
histV = cv.calcHist([imgPrueba], [2], None, [256], [0, 256])
histV = histV/max(histV)

plt.figure('H')
plt.figure('S')
plt.figure('V')


plt.figure('H')
line1=plt.plot(list(range(180)),histH)
plt.show()

plt.figure('S')
line2=plt.scatter(list(range(256)),histS)
plt.show()

plt.figure('V')
line3=plt.scatter(list(range(256)),histV)
plt.show()


# #Codigo para poder visualizar el video. Se agrega la linea.
# cont = 0
# prom = 0

# kernelPromedio = np.ones((3,3),np.float32)/9
# while (video.isOpened()):
#   ret, img = video.read()
#   if ret == True:
      
#       dimensionesImg = img.shape

#       w = dimensionesImg[1]
#       H = dimensionesImg[0]
      
#       cont = cont + 1
#       if cont == 20:
#           imgPrueba = cv.GaussianBlur(img,(5,5),0)
#           imgPrueba = cv.pyrMeanShiftFiltering(imgPrueba,sp=15,sr=40,maxLevel = 3)
#           bordes = cv.Canny(imgPrueba,250,400)
#           tamaño = np.asarray(bordes)
#           dame = np.array(np.where(tamaño == 255))
#           prom = int(np.mean(dame[0,:]))
          
#           cont = 0
      
#       cv.line(img,(0,prom),(w,prom),(255,0,0))
#       cv.imshow('video', imgPrueba)
#       #Se oprime s para escapar del video.
#       if cv.waitKey(30) == ord('s'):
#           break
#   else: break
      
# video.release()
# cv.destroyAllWindows()
