from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_train_frame = "D:/RosBag/dataSet/train_frames"
path_train_mask = "D:/RosBag/dataSet/train_masks"

path_val_frame = "D:/RosBag/dataSet/val_frames"
path_val_mask = "D:/RosBag/dataSet/val_masks"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.5,0.95],
    horizontal_flip=True,
    brightness_range=[0.5,1.2],
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_image_generator = train_datagen.flow_from_directory(
path_train_frame,
batch_size = 32,
save_to_dir=path_train_frame)

train_mask_generator = train_datagen.flow_from_directory(
path_train_mask,
batch_size = 32,
save_to_dir=path_train_mask)

val_image_generator = val_datagen.flow_from_directory(
path_val_frame,
batch_size = 32)


val_mask_generator = val_datagen.flow_from_directory(
path_val_mask,
batch_size = 32)



train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)
