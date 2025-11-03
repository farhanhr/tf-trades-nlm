from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_training_data(dataset_path, batch_size=16):
    train_dir = os.path.join(dataset_path, 'train')
    print("Loading training data from:", train_dir)
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        brightness_range=[0.90, 1.10],
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=0.1
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        subset='training',
        class_mode='sparse'
    )

    val_datagen = ImageDataGenerator(validation_split=0.1)
    val_gen = val_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen
