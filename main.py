import tensorflow as tf
from train.trades_train import trades_training_loop

if __name__ == "__main__":
    print("🧠 TensorFlow version:", tf.__version__)
    print("📦 Devices detected:")
    print(tf.config.list_physical_devices())

    if tf.config.list_physical_devices('GPU'):
        print("GPU will be used for training.")
    else:
        print("No GPU detected. CPU will be used.")

        
    trades_training_loop()
