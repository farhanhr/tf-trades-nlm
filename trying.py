import tensorflow as tf
from configs.train_config import config
from data.data_loader import get_training_data
from models.vgg_models import load_models

train_gen, val_gen = get_training_data(config["dataset_path"], config["batch_size"])

import matplotlib.pyplot as plt
import numpy as np

# Ambil satu batch dari training generator
X_batch, y_batch = next(train_gen)

print("Shape X_batch:", X_batch.shape)
print("Shape y_batch:", y_batch.shape)
print("y_batch example:", y_batch[:10])  # tampilkan 10 label pertama

# Tampilkan beberapa gambar dari batch
num_images_to_show = 5
plt.figure(figsize=(15,3))
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i+1)
    plt.imshow(X_batch[i].astype('uint8'))
    if y_batch.ndim == 2 and y_batch.shape[1] == 1:
        label = int(y_batch[i])
    elif y_batch.ndim == 1:
        label = int(y_batch[i])
    else:
        label = y_batch[i]
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.show()
