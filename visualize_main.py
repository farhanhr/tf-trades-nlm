import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.load_models import load_models
from inference.attacks import generate_adversarial_examples_trades, generate_adversarial_examples_PGD
from inference.visualize_image import visualize_preprocessing_and_attack
from configs.train_config import config
from data.preprocessors import compose_preprocessors, bilateral_denoise_tf, total_variation_denoise_tf, nlm_denoise_bpda
from data.data_loader import get_training_data
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

saved_models_path = config["saved_models_path"] + "BrainTumorMRI_VGG16_03112025_2358_model.h5"
model = load_models(saved_models_path, compile=True)

preprocess_fn = compose_preprocessors([
bilateral_denoise_tf(spatial_sigma=1.0, intensity_sigma=25.0, kernel_size=3),
# total_variation_denoise_tf(weight=1.3, iterations=100),
# nlm_denoise_bpda(h=0.2, patch_size=3, window_size=5),
# vgg_preprocess
])

_, val_gen = get_training_data(config["dataset_path"], 8, preprocessing_function=None)
best_val_loss, wait, patience = float("inf"), 0, 10

epochs = config["epochs"]
step_size = config["step_size"]
epsilon = config["epsilon"]
perturb_steps = config["perturb_steps"]
beta = config["beta"]

visualize_preprocessing_and_attack(model, preprocess_fn, 4, 1)

# x_batch_raw, y_batch = next(iter(val_gen))
# x_batch_raw = tf.cast(x_batch_raw, tf.float32)
# x_adv = generate_adversarial_examples_PGD(model, x_batch_raw, epsilon, step_size, perturb_steps)


# x_adv = preprocess_fn(x_adv)
# idx = 0
# x_adv_sample = x_adv[idx]  # (H, W, C)
# x_adv_sample = tf.clip_by_value(x_adv_sample, 0, 255)
# x_adv_sample = tf.cast(x_adv_sample, tf.uint8)  

# plt.figure(figsize=(224,224))
# plt.imshow(x_adv_sample.numpy()) 
# plt.axis('off')
# plt.title("Pure Adversarial Example (x_adv)")
# plt.show()