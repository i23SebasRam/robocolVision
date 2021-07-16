import keras_segmentation as ks
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping

#Paths
Path_Train_Frames = "D:/RosBag/dataSet/train_frames/new/"
Path_Train_Masks = "D:/RosBag/dataSet/train_masks/new/"

Path_Val_Frames = "D:/RosBag/dataSet/val_frames/new/"
Path_val_Masks = "D:/RosBag/dataSet/val_masks/new/" 

Path_test_Frames = "D:/RosBag/dataSet/test_frames/new"
Path_test_Masks = "D:/RosBag/dataSet/test_masks/new"

weights_path = "D:/RosBag/dataSet/Modelo"
path_csv = "D:/RosBag/dataSet/Modelo/info.csv"

#Parameters
No_Training_img = len(os.listdir(Path_Train_Frames))
No_Epochs = 20
Batch_Size = 4
Batch_Size_val = 4
img_size = (224,224)

#Path list images
train_frames_paths = [
        os.path.join(Path_Train_Frames, fname)
        for fname in os.listdir(Path_Train_Frames)
    ]

train_masks_paths = [
        os.path.join(Path_Train_Masks, fname)
        for fname in os.listdir(Path_Train_Masks)
    ]

val_frames_paths = [
        os.path.join(Path_Val_Frames, fname)
        for fname in os.listdir(Path_Val_Frames)
    ]

val_masks_paths = [
        os.path.join(Path_val_Masks, fname)
        for fname in os.listdir(Path_val_Masks)
    ]

#Class for the images
class images(Sequence):
    def __init__(self, batch_size, img_size, input_img_paths,mask_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.mask_img_paths = mask_img_paths
    
    def __len__(self):
        return len(self.mask_img_paths) // self.batch_size
    
    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i:i + self.batch_size]
        batch_mask_img_paths = self.mask_img_paths[i:i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype = "float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img)
            x[j] = img/255
        y = np.zeros((self.batch_size,) + (50176,) + (3,), dtype = "float32")
        for j, path in enumerate(batch_mask_img_paths):
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img).reshape((50176,3))
            y[j] = img/255
        
        return x,y

#Model
img_input = keras.Input((224,224,3))

conv1 = layers.Conv2D(32,(3,3),activation='relu', padding='same')(img_input)
conv1 = layers.Dropout(0.2)(conv1)
conv1 = layers.Conv2D(32,(3,3),activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2,2))(conv1)

conv2 = layers.Conv2D(64,(3,3),activation='relu', padding='same')(pool1)
conv2 = layers.Dropout(0.2)(conv2)
conv2 = layers.Conv2D(64,(3,3),activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2,2))(conv2)

conv3 = layers.Conv2D(128,(3,3),activation='relu', padding='same')(pool2)
conv3 = layers.Dropout(0.2)(conv3)
conv3 = layers.Conv2D(128,(3,3),activation='relu', padding='same')(conv3)

up1 = concatenate([layers.UpSampling2D((2,2))(conv3),conv2], axis=-1)

conv4 = layers.Conv2D(64,(3,3),activation='relu', padding='same')(up1)
conv4 = layers.Dropout(0.2)(conv4)
conv4 = layers.Conv2D(64,(3,3),activation='relu', padding='same')(conv4)

up2 = concatenate([layers.UpSampling2D((2,2))(conv4),conv1], axis=-1)

conv5 = layers.Conv2D(32,(3,3),activation='relu', padding='same')(up2)
conv5 = layers.Dropout(0.2)(conv5)
conv5 = layers.Conv2D(32,(3,3),activation='relu', padding='same')(conv5)

output = layers.Conv2D(3,(1,1), padding='same')(conv5)


model = keras.Model(img_input,output)

model.summary()

#Callbacks
checkpoint = ModelCheckpoint(weights_path,save_best_only=True)
csv_logger = CSVLogger(path_csv,separator=';',append=True)
earlyStopping = EarlyStopping(min_delta=0.01,patience=3)

callbacks_list = [checkpoint,csv_logger,earlyStopping]

#Batch organized images
train_gen = images(Batch_Size, img_size, train_frames_paths, train_masks_paths)
val_gen = images(Batch_Size_val, img_size, val_frames_paths, val_masks_paths)


#Model compile and fit
model.compile(
    optimizer = 'rmsprop',
    loss = 'categorical_crossentropy',
    metrics = [keras.metrics.BinaryAccuracy()]
)

model.fit(train_gen,epochs=No_Epochs,validation_data=val_gen, callbacks=callbacks_list)