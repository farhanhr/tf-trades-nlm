import tensorflow as tf

def load_models(saved_models_path, compile=False, trainable=False):
    model = tf.keras.models.load_model(saved_models_path, compile=compile)
    if trainable == True:
        # Unfreeze 10 last layers
        for layer in model.layers[-10:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                print("[INFO] Unfroze the layers of VGG16.")
    return model