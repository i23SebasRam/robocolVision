import os
import cv2 as cv
import numpy as np

#Datapaths
generalPath = 'D:/RosBag/dataSet/'
maskPath = generalPath + 'mask/'
maskFixedPath = generalPath + 'maskFixed/'
testImagePath = maskPath + 'img013011.png'

#Names of the images
namesMask = os.listdir(maskPath)
namesMask = namesMask
print(testImagePath)

#Note*
"""
Label -- Hue value -- value
Rocks       102        0
Terrain       0        1
People      134        2
Building     30        3
Robot        44        4
Sky          89        5
Trees        78        6
Station S   112        7
What ever    45        8
"""

listLabel = np.array([102,0,134,30,44,89,78,112,45],dtype=np.uint8)

#Identify the value pixels
def identifyPixels(img,list):
    img = img[:,:,0]
    for idx,value in enumerate(list):
        img = np.where(img == value, idx ,img)
    return img

#Get and save the images
for i in namesMask:
    img = cv.imread(maskPath + i)
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    img = identifyPixels(img,listLabel)
    os.chdir(maskFixedPath)
    cv.imwrite(i,img)

