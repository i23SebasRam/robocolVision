from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_train_frame = "D:/RosBag/dataSet/train_frames"
path_train_mask = "D:/RosBag/dataSet/train_masks"

path_val_frame = "D:/RosBag/dataSet/val_frames"
path_val_mask = "D:/RosBag/dataSet/val_masks"

path_test_frame = "D:/RosBag/dataSet/test_frames"
path_test_mask = "D:/RosBag/dataSet/test_masks"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.5,0.95],
    horizontal_flip=True,
    brightness_range=[0.5,1.2],
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_image_generator = train_datagen.flow_from_directory(
path_train_frame,
target_size=(224,224),
batch_size = 1,
save_to_dir=path_train_frame + '/new/',
seed=41)

train_mask_generator = train_datagen.flow_from_directory(
path_train_mask,
target_size=(224,224),
batch_size = 1,
save_to_dir=path_train_mask + '/new/',
seed=41)

val_image_generator = val_datagen.flow_from_directory(
path_val_frame,
target_size=(224,224),
batch_size = 1,
save_to_dir= path_val_frame + '/new/',
seed = 41)


val_mask_generator = val_datagen.flow_from_directory(
path_val_mask,
target_size=(224,224),
batch_size = 1,
save_to_dir= path_val_mask + '/new/',
seed=41)

test_image_generator = test_datagen.flow_from_directory(
    path_test_frame,
    target_size=(224,224),
    batch_size=1,
    save_to_dir= path_test_frame + '/new/',
    seed=41)

test_mask_generator = test_datagen.flow_from_directory(
    path_test_mask,
    target_size=(224,224),
    batch_size=1,
    save_to_dir= path_test_mask + '/new/',
    seed=41
)

for i in range(132):
    image = next(train_image_generator)
    image2 = next(train_mask_generator)

#This both below only need to scale the pixels, so just scale the amount there is.
for i in range(19):
    image = next(val_image_generator)
    image = next(val_mask_generator)

for i in range(10):
    image = next(test_image_generator)
    image = next(test_mask_generator)


