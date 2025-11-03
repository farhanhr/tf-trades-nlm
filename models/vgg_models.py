import tensorflow as tf
import os 

def load_models(saved_models_path, compile=False):
    model = tf.keras.models.load_model(saved_models_path, compile=compile)
    return model