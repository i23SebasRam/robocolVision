from numpy import expand_dims
import numpy as np
from numpy.core.fromnumeric import shape, size
from numpy.random import seed
from scipy.ndimage.interpolation import shift
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

#Horizontal and vertical shift
def Shift(imagex,imagey,number,post,showImage = False,widthShift = 0.05,heightShift = 0.05):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(width_shift_range=widthShift,height_shift_range=heightShift,fill_mode="wrap")
    itx = datagen.flow(samplesx,batch_size=1,seed=42,save_to_dir="D:/testAugmen/xAugmented",save_prefix='x%s'%post)
    ity = datagen.flow(samplesy,batch_size=1,seed=42,save_to_dir="D:/testAugmen/yAugmented",save_prefix='y%s'%post)
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
#Horizontal and vertical flip
def flip(imagex,imagey,number,post,showImage = False, horizontalFlip = True, verticalFlip = True, rotation = 10):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(horizontal_flip = horizontalFlip, vertical_flip = verticalFlip, rotation_range= rotation)
    itx = datagen.flow(samplesx,batch_size=1,seed=42,save_to_dir="D:/testAugmen/xAugmented",save_prefix='x%s'%post)
    ity = datagen.flow(samplesy,batch_size=1,seed=42,save_to_dir="D:/testAugmen/yAugmented",save_prefix='y%s'%post)
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
#Change the brightness
def brightness(imagex,imagey,number,post,showImage = False, rotation = False,rotationRange = 5):
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
    itx = datagen.flow(samplesx,batch_size=1,seed=42,save_to_dir="D:/testAugmen/xAugmented",save_prefix='x%s'%post)
    ity = datageny.flow(samplesy,batch_size=1,seed=42,save_to_dir="D:/testAugmen/yAugmented",save_prefix='y%s'%post)
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
#Do zoom in or zoom out
def zoom(imagex,imagey,number,post,showImage = False, zoomRange = [0.5,1.5], zoomIn = False, zoomOut=False):
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
    itx = datagen.flow(samplesx,batch_size=1,seed=42,save_to_dir="D:/testAugmen/xAugmented",save_prefix='x%s'%post)
    ity = datageny.flow(samplesy,batch_size=1,seed=42,save_to_dir="D:/testAugmen/yAugmented",save_prefix='y%s'%post)
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
#All the augmentations
def generalAugmentation(imagex,imagey,number,post,showImage = False,horizontalFlip=False,verticalFlip=False,zoomRange=[1,1],bright = [1,1],rotation = 0):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(horizontal_flip=horizontalFlip,vertical_flip=verticalFlip,zoom_range=zoomRange,brightness_range=bright,rotation_range=rotation,fill_mode="wrap")
    datageny = ImageDataGenerator(horizontal_flip=horizontalFlip,vertical_flip=verticalFlip,zoom_range=zoomRange,rotation_range=rotation,fill_mode="wrap")
    itx = datagen.flow(samplesx,batch_size=1,shuffle=False,save_to_dir="D:/testAugmen/xAugmented",save_prefix='x%s'%post)
    ity = datageny.flow(samplesy,batch_size=1,shuffle=False,save_to_dir="D:/testAugmen/yAugmented",save_prefix='y%s'%post)
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
#Load the image from a file
def loadImage(listNameX,listNameY,showImageO = False,numberI=5,mode = "general_augmentation"):
    listName = np.array([listNameX,listNameY])
    idx = listName.shape
    for i in range(idx[1]):
        rutax = os.path.join("D:\\testAugmen","x",listName[0,i])
        rutay = os.path.join("D:\\testAugmen","y",listName[1,i])
        imgx = load_img(rutax)
        imgy = load_img(rutay)
        if mode == "shift":
            Shift(imgx,imgy,number=numberI,showImage = showImageO,post=i)
        elif mode == "shift":
            flip(imgx,imgy,number=numberI,showImage = showImageO,post=i)
        elif mode == "shift":
            brightness(imgx,imgy,number=numberI,showImage = showImageO,post=i)
        elif mode == "shift":
            zoom(imgx,imgy,number=numberI,showImage = showImageO,post=i)
        else:
            generalAugmentation(
            imgx,
            imgy,
            number=numberI,
            horizontalFlip=True,
            verticalFlip=True,
            zoomRange=[0.5,0.95], 
            bright=[0.5,1.5],
            rotation=5,
            showImage = showImageO,
            post=i
            )
            
loadImage(listNamesX,listNamesY,numberI=5)




