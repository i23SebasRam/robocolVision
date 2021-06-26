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

def horVerShift(imagex,imagey,number):
    datax = img_to_array(imagex)
    datay = img_to_array(imagey)
    samplesx = expand_dims(datax,0)
    samplesy = expand_dims(datay,0)
    datagen = ImageDataGenerator(width_shift_range=0.05,height_shift_range=0.05)
    itx = datagen.flow(samplesx,batch_size=1,seed=42)
    ity = datagen.flow(samplesy,batch_size=1,seed=42)
    for i in range(number):
        #pyplot.subplot(330+1+i)
        batchx = itx.next()
        batchy = ity.next()
        imagex = batchx[0].astype('uint8')
        imagey = batchy[0].astype('uint8')
        #pyplot.imshow(imagey)
        #pyplot.imshow(imagex)
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
