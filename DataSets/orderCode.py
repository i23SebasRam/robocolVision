import os 
import random
import re
from PIL import Image

Data_Path = "D:/RosBag/dataSet/"
Frame_Path = Data_Path + "frames/"
Mask_Path = Data_Path + "mask/"

folders = ['train_frames','train_masks','val_frames','val_masks','test_frames','test_masks']

for folder in folders:
    os.makedirs(Data_Path + folder)

all_frames = os.listdir(Frame_Path)
all_masks = os.listdir(Mask_Path)

all_frames.sort(key=lambda var:[int(x) if x.isdigit() else x
                            for x in re.findall(r'[^0-9]|[0-9]+', var)])



all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                            for x in re.findall(r'[^0-9]|[0-9]+', var)])


random.seed(230)
random.shuffle(all_frames)


train_split = int(0.7*len(all_frames))
val_split = int(0.9*len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]
print(test_frames)

#train_masks = [f for f in all_masks if f[0:9]+'.jpg' in train_frames]
#val_masks = [f for f in all_masks if f[0:9]+'.jpg' in val_frames]
#test_masks = [f for f in all_masks if f[0:9]+'.jpg' in test_frames]

train_masks = [f for f in all_masks if f in train_frames]
val_masks = [f for f in all_masks if f in val_frames]
test_masks = [f for f in all_masks if f in test_frames]



def add_frames(dir_name, image):
  
  img = Image.open(Frame_Path + image)
  img.save(Data_Path +'/{}'.format(dir_name)+'/'+image)
  
  
  
def add_masks(dir_name, image):
  
  img = Image.open(Mask_Path + image)
  img.save(Data_Path +'/{}'.format(dir_name)+'/'+image)


frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                 (test_frames, 'test_frames')]

mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                (test_masks, 'test_masks')]

# Add frames

for folder in frame_folders:
  
  array = folder[0]
  name = [folder[1]] * len(array)

  list(map(add_frames, name, array))
         
    
# Add masks

for folder in mask_folders:
  
  array = folder[0]
  name = [folder[1]] * len(array)
  
  list(map(add_masks, name, array))