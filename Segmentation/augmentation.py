from numpy import expand_dims
import numpy as np
from numpy.core.fromnumeric import shape, size
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os

from tensorflow.python.ops.gen_batch_ops import batch

#Path of Working directory
pathx = 'D:/testAugmen/x'
pathy = 'D:/testAugmen/y'

#Take the files
listNamesX = os.listdir(pathx)
listNamesY = os.listdir(pathy)
ruta = os.path.join("D:\\testAugmen","x",listNamesX[0])
img = load_img(ruta)
#pyplot.imshow(img)
#pyplot.show()
#print(size(listNamesX))

#Load the images

#def loadImage(listNamesx,listNamesy,rutax,rutay):
    #npX = np.array([])
    #npY = np.array([])
    #for idx,item in enumerate(listNamesx):
    #    pathx = os.path.join(rutax,item)
    #    img = load_img(pathx)
    #    npX[:,:,:,idx] = img
    
   # for idy,item in enumerate(listNamesy):
   #     pathy = os.path.join(rutay,item)
   #     img = load_img(pathx)
   #     npY[:,:,:,idy] = img
   # return npX,npY 

#Path of Working directory
pathx = 'D:/testAugmen/x'
pathy = 'D:/testAugmen/y'

#Take the files
rutax = ["D:\\testAugmen","x"]
rutay = ["D:\\testAugmen","y"]
listNamesX = os.listdir(pathx)
listNamesY = os.listdir(pathy)

#imgX,imgY = load_img(listNamesX,listNamesY,pathx,pathy)

#print(imgX.shape())

#Horizontal or vertical shift.

def horVerShift(imagex,imagey,number,showImage = False,widthShift = 0.05,heightShift = 0.05):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(width_shift_range=widthShift,height_shift_range=heightShift)
    itx = datagen.flow(samplesx,batch_size=1,seed=42)
    ity = datagen.flow(samplesy,batch_size=1,seed=42)
    for i in range(number):
        batchx = itx.next()
        batchy = ity.next()
        imagex = batchx[0].astype('uint8')
        imagey = batchy[0].astype('uint8')
        if showImage == True:
            #pyplot.subplot(330+1+i)
            #pyplot.imshow(imagey)
            #pyplot.imshow(imagex)
            fig = pyplot.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax1.imshow(imagex)
            ax2.imshow(imagey)
    pyplot.show()

def flip(imagex,imagey,number,showImage = False, horizontalFlip = True, verticalFlip = True, rotation = 10):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(horizontal_flip = horizontalFlip, vertical_flip = verticalFlip, rotation_range= rotation)
    itx = datagen.flow(samplesx,batch_size=1,seed=42)
    ity = datagen.flow(samplesy,batch_size=1,seed=42)
    for i in range(number):
        batchx = itx.next()
        batchy = ity.next()
        imagex = batchx[0].astype('uint8')
        imagey = batchy[0].astype('uint8')
        if showImage == True:
            fig = pyplot.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax1.imshow(imagex)
            ax2.imshow(imagey)
    pyplot.show()


def brightness (imagex,imagey,number,showImage = False, rotation = False,rotationRange = 5):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(brightness_range=[0.5,1.5])
    if rotation == True:
        datagen = ImageDataGenerator(brightness_range=[0.5,1.5],rotation_range=rotationRange)
    datageny = ImageDataGenerator()
    #Range less - 1->Darken image. 
    #More - 1 -> Brighten image.
    itx = datagen.flow(samplesx,batch_size=1,seed=42)
    ity = datageny.flow(samplesy,batch_size=1,seed=42)
    for i in range(number):
        batchx = itx.next()
        batchy = ity.next()
        imagex = batchx[0].astype('uint8')
        imagey = batchy[0].astype('uint8')
        if showImage == True:
            fig = pyplot.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax1.imshow(imagex)
            ax2.imshow(imagey)
    pyplot.show()


def zoom(imagex,imagey,number,showImage = False, zoomRange = [0.5,1.5], zoomIn = False, zoomOut=False):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(zoom_range=zoomRange)
    datageny = ImageDataGenerator(zoom_range=zoomRange)
    if zoomIn == True:
        datagen = ImageDataGenerator(zoom_range=[0.5,0.95])
        datageny = ImageDataGenerator(zoom_range=[0.5,0.95])
    if zoomOut == True:
        datagen = ImageDataGenerator(zoom_range=[1.05,1.5])
        datageny = ImageDataGenerator(zoom_range=[1.05,1.5])  
    #Range less - 1->Zoom in. 
    #More - 1 -> Zoom out.
    itx = datagen.flow(samplesx,batch_size=1,seed=42)
    ity = datageny.flow(samplesy,batch_size=1,seed=42)
    for i in range(number):
        batchx = itx.next()
        batchy = ity.next()
        imagex = batchx[0].astype('uint8')
        imagey = batchy[0].astype('uint8')
        if showImage == True:
            fig = pyplot.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax1.imshow(imagex)
            ax2.imshow(imagey)
    pyplot.show()


rutax = os.path.join("D:\\testAugmen","x",listNamesX[0])
rutay = os.path.join("D:\\testAugmen","y",listNamesY[0])
imgx = load_img(rutax)
imgy = load_img(rutay)
horVerShift(imgx,imgy,9)
flip(imgx,imgy,9)
brightness(imgx,imgy,9)
zoom(imgx,imgy,9,showImage=True,zoomOut=True)





