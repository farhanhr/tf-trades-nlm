from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

def get_training_data(
    dataset_path, 
    batch_size=32, 
    validation_split=0.2, 
    target_size=(224, 224), 
    preprocessing_function=None
):
    train_dir = os.path.join(dataset_path, 'train')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        brightness_range=[0.90, 1.10],
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split
    )
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset='training',
        class_mode='sparse',
        shuffle=True
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        validation_split=validation_split
    )
    val_gen = val_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen


def get_testing_data(dataset_path, batch_size=32, preprocessing_fn=None):
    test_dir = os.path.join(dataset_path, 'test')
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
        rescale=None
        )
    test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=False
    )
    return test_gen

def get_adversarial_data(dataset_path, batch_size=32, preprocessing_fn=None):
    test_dir = os.path.join(dataset_path, 'Adversarial_Example/Testing_FGSM_Adversarial_Example')
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
        rescale=None
        )
    test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=False
    )
    return test_gen