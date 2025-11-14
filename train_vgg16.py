import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from configs.train_config import config
from data.data_loader import get_training_data
from models.vgg16_models import create_model_vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

model = create_model_vgg16()
model.compile(optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

train_gen, val_gen = get_training_data(
    config["dataset_path"],
    config["batch_size"],
    validation_split=0.2,
    preprocessing_function=vgg_preprocess
)

labels = train_gen.classes
classes = np.unique(labels)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=labels
)

class_weights_dict = dict(zip(classes, weights))

print("Bobot Kelas yang Dihitung (di mana nilai lebih tinggi = kelas minoritas):")
print(class_weights_dict)

saved_model_dir = config["saved_models_path"]
os.makedirs(saved_model_dir, exist_ok=True)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
saved_model_path = os.path.join(saved_model_dir, 'BrainTumorMRI_VGG16_03112025_2358_model.h5')

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.30,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    saved_model_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

trained_model = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)


plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(trained_model.history['accuracy'], label='Train Accuracy')
plt.plot(trained_model.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(trained_model.history['loss'], label='Train Loss')
plt.plot(trained_model.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.show()
