from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.applications import ResNet50V2
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dtype, softmax
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img

#Paths
Path_Train_Frames = "D:/RosBag/dataSet/train_frames/new"
Path_Train_Masks = "D:/RosBag/dataSet/train_masks/new"

Path_Val_Frames = "D:/RosBag/dataSet/val_frames/new"
Path_val_Masks = "D:/RosBag/dataSet/val_masks/new" 

Path_test_Frames = "D:/RosBag/dataSet/test_frames/new"
Path_test_Masks = "D:/RosBag/dataSet/test_masks/new"

weights_path = "D:/RosBag/dataSet/Modelo"
path_csv = "D:/RosBag/dataSet/Modelo/info.csv"

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

        x = np.zeros((self.batch_size,)) + self.img_size + (3,), dtype = "float32"
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,)) + self.img_size + (3,), dtype = "float32"
        for j, path in enumerate(batch_mask_img_paths):
            img = load_img(path, target_size=self.img_size)
            y[j] = img
        
        return x,y

#Parameters
No_Training_img = len(os.listdir(Path_Train_Frames))
No_Epochs = 30
Batch_Size = 20
Batch_Size_val = 2
img_size = (224,224)


#Base Model
base_model = ResNet50V2(weights='imagenet',
                        include_top=False,
                        input_shape=(224,224,3))
base_model.trainable = False

#Callbacks
checkpoint = ModelCheckpoint(weights_path,save_best_only=True)
csv_logger = CSVLogger(path_csv,separator=';',append=True)
earlyStopping = EarlyStopping(min_delta=0.01,patience=3)

callbacks_list = [checkpoint,csv_logger,earlyStopping]
#Top Model
input = keras.Input(shape=(224,224,3))

#Inferior of the model
x = base_model(input,training = False)

for filters in [2048,1024,512,256,128,64]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters,3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    if filters != 64:
        x = layers.UpSampling2D(2)(x)


output = layers.Conv2D(3,9,activation="softmax",padding="same")(x)

model = keras.Model(input,output)

model.summary()

#Batch organized images
train_gen = images(Batch_Size, img_size, Path_Train_Frames, Path_Train_Masks)
val_gen = images(Batch_Size_val, img_size, Path_Val_Frames, Path_val_Masks)

#Model compile and fit

model.compile(
    optimizer = 'rmsprop',
    loss = 'sparse_categorical_crossentropy',
    metrics = [keras.metrics.BinaryAccuracy()]
)

model.fit(train_gen,epochs=No_Epochs,validation_data=val_gen, callbacks=callbacks_list)



