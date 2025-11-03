import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tqdm import tqdm
from configs.train_config import config
from data.preprocessors import preprocess_with_nlm_gpu
from data.data_loader import get_training_data
from models.saved_models import load_models
from models.trades_loss import trades_loss_for_vgg

def trades_training_loop():
    train_gen, val_gen = get_training_data(config["dataset_path"], config["batch_size"])
    model = load_models(compile=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    preprocess_fn = lambda x: preprocess_with_nlm_gpu(
        x,
        use_nlm=True,
        h=10.0,
        patch_size=3,
        window_size=7,
        vgg_preprocess_fn=vgg_preprocess
    )

    best_val_loss, wait, patience = float("inf"), 0, 3
    epochs = config["epochs"]
    step_size = config["step_size"]
    epsilon = config["epsilon"]
    perturb_steps = config["perturb_steps"]
    beta = config["beta"]
    val_subset_size = config["val_subset_size"]
    checkpoint_TRADES_model_dir = config["checkpoint_dir"] + "[Best_checkpoint]TRADES_model_03112025_2010.keras"
    final_TRADES_model_dir = config["final_model_dir"] + "[Final]TRADES_model_03112025_2010.keras"


    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        batch_losses = []

        train_pbar = tqdm(enumerate(train_gen), total=len(train_gen), desc="Training", ncols=100)

        for x_batch_raw, y_batch in train_pbar:
            x_batch_raw = tf.cast(x_batch_raw, tf.float32)

            loss = trades_loss_for_vgg(
                model=model,
                x_natural_raw=x_batch_raw,
                y=y_batch,
                optimizer=optimizer,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                beta=beta,
                distance="l_inf",
                preprocess_fn=preprocess_fn
            )

            batch_losses.append(float(loss.numpy()))

            train_pbar.set_postfix({"loss": f"{np.mean(batch_losses[-10:]):.4f}"})

        train_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1} mean training loss: {train_loss:.4f}")

        val_losses = []
        val_sample_count = 0

        val_pbar = tqdm(enumerate(val_gen), total=min(len(val_gen), val_subset_size // val_gen.batch_size), desc="Validation", ncols=100)

        for x_val, y_val in val_pbar:
            x_val = tf.cast(x_val, tf.float32)
            preds = model(preprocess_fn(x_val), training=False)
            val_loss = tf.keras.losses.categorical_crossentropy(y_val, preds)
            val_losses.append(np.mean(val_loss.numpy()))

            val_sample_count += len(x_val)
            val_pbar.set_postfix({"val_loss": f"{np.mean(val_losses):.4f}"})

            if val_sample_count >= val_subset_size:
                break

        val_loss_mean = np.mean(val_losses)
        print(f"Validation loss (sampled): {val_loss_mean:.4f}")

        if val_loss_mean < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} → {val_loss_mean:.4f}")
            best_val_loss = val_loss_mean
            wait = 0
            model.save(checkpoint_TRADES_model_dir)
            print(f"Checkpoint saved at: {checkpoint_TRADES_model_dir}")
        else:
            wait += 1
            print(f"No improvement from {best_val_loss:.4f}. Patience counter: {wait}/{patience}")

        if wait >= patience:
            print("Early stopping triggered — no improvement for several epochs.")
            break

    print(f"\nTraining done. Saved final model to: {final_TRADES_model_dir}")
    model.save(final_TRADES_model_dir)

