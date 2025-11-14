from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from typing import Literal, Callable
import os
import tensorflow as tf
import numpy as np

class PreprocessedDirectoryIterator(Sequence):
    """Membungkus DirectoryIterator agar batch diproses dengan fungsi preprocess kustom."""
    def __init__(self, base_generator, preprocess_fn: Callable[[tf.Tensor], tf.Tensor]):
        self.base_generator = base_generator
        self.preprocess_fn = preprocess_fn

        # salin atribut penting agar kompatibel dengan evaluate/predict
        self.n = base_generator.n
        self.batch_size = base_generator.batch_size
        self.class_indices = base_generator.class_indices
        self.classes = base_generator.classes
        self.filenames = base_generator.filenames
        self.num_classes = getattr(base_generator, "num_classes", None)

    def __len__(self):
        # jumlah batch total
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        # ambil batch ke-idx dari base generator
        batch_x, batch_y = self.base_generator[idx]

        # lakukan preprocess untuk setiap gambar (vektorisasi via tf.map_fn)
        # preprocess_fn diharapkan menerima tensor dengan shape (1, H, W, C) atau (H,W,C)
        batch_x = tf.map_fn(
            lambda img: tf.squeeze(self.preprocess_fn(tf.expand_dims(img, 0)), axis=0),
            batch_x,
            fn_output_signature=tf.float32
        )

        return batch_x, batch_y

    def on_epoch_end(self):
        # sinkronisasi ulang jika base_generator diacak
        if hasattr(self.base_generator, "on_epoch_end"):
            self.base_generator.on_epoch_end()


def get_training_data(
    dataset_path,
    batch_size=32,
    validation_split=0.2,
    target_size=(224, 224),
    preprocessing_function=None
):
    """
    Mengembalikan (train_gen, val_gen).
    IMPORTANT: jika model VGG16 pretrained digunakan, berikan preprocessing_function=vgg_preprocess
    (yang mengharapkan input dalam skala 0-255).
    """
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
    """
    Mengembalikan generator test.
    Jangan gunakan rescale=1./255 di sini jika model mengharapkan 0-255.
    Jika preprocessing_fn diberikan (mis. vgg_preprocess), bungkus test_gen agar preprocessing diterapkan.
    """
    test_dir = os.path.join(dataset_path, 'test')
    test_datagen = ImageDataGenerator(
        preprocessing_function=None  # kita terapkan preprocess_fn via wrapper bila diperlukan
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        color_mode='rgb',
        shuffle=False
    )

    if preprocessing_fn is not None:
        test_gen = PreprocessedDirectoryIterator(test_gen, preprocessing_fn)

    return test_gen


def get_adversarial_data(
    dataset_path,
    batch_size=32,
    preprocessing_fn=None,
    type: Literal["FGSM", "PGD"] = "FGSM"
):
    """
    Mengembalikan generator untuk adversarial examples yang disimpan pada folder:
      Adversarial_Example/FGSM_example atau Adversarial_Example/PGD_example
    """
    attack_type = "PGD_example" if type == "PGD" else "FGSM_example"
    attack_type_path = os.path.join("Adversarial_Example", attack_type)
    test_dir = os.path.join(dataset_path, attack_type_path)

    test_datagen = ImageDataGenerator(
        preprocessing_function=None
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        color_mode='rgb',
        shuffle=False
    )

    if preprocessing_fn is not None:
        test_gen = PreprocessedDirectoryIterator(test_gen, preprocessing_fn)

    return test_gen
