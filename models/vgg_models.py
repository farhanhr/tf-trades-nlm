import tensorflow as tf

def load_models(compile=False):
    saved_model = 'saved_models/BrainTumorMRI_VGG16_26102025_0850_model.keras'
    model = tf.keras.models.load_model(saved_model, compile=compile)
    return model