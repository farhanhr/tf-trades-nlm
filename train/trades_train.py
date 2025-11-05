import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from tqdm import tqdm
from configs.train_config import config
from data.data_loader import get_training_data
from data.preprocessors import preprocess_with_nlm_gpu
from models.load_models import load_models
from models.trades_loss import trades_loss_for_vgg


def trades_training_loop():
    saved_models_path = config["saved_models_path"] + "BrainTumorMRI_VGG16_03112025_2358_model.h5"
    model = load_models(saved_models_path, compile=False)

    # Learning Rate Schedule
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=20,   
        t_mul=2.0,             
        m_mul=0.8,              
        alpha=1e-5              
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    preprocess_fn = lambda x: preprocess_with_nlm_gpu(
        x,
        use_nlm=True,
        h=0.15,
        patch_size=3,
        window_size=5,
        vgg_preprocess_fn=vgg_preprocess
    )
    
    train_gen, val_gen = get_training_data(config["dataset_path"], 8)
    best_val_loss, wait, patience = float("inf"), 0, 10
    epochs = config["epochs"]
    step_size = config["step_size"]
    epsilon = config["epsilon"]
    perturb_steps = config["perturb_steps"]
    beta = config["beta"]

    checkpoint_TRADES_model_dir = config["checkpoint_dir"] + "[Best_checkpoint]TRADES_model_04112025_2235.h5"
    final_TRADES_model_dir = config["final_model_dir"] + "[Final]TRADES_model_03112025_2235.h5"

    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} — LR: {optimizer.learning_rate(epoch).numpy():.6f}")
        batch_losses = []

        train_pbar = tqdm(enumerate(train_gen), total=len(train_gen), desc="Training", ncols=100)
        for step, (x_batch_raw, y_batch) in train_pbar:
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
            if (step + 1) % 10 == 0:
                train_pbar.set_postfix({"loss": f"{np.mean(batch_losses[-10:]):.4f}"})

            if step + 1 >= len(train_gen):
                break

        train_loss = np.mean(batch_losses)
        train_loss_history.append(train_loss)
        print(f"Epoch {epoch+1} mean training loss: {train_loss:.4f}")

        # Validation
        val_losses = []
        val_pbar = tqdm(range(len(val_gen)), desc="Validation", ncols=100)
        for step in val_pbar:
            x_val, y_val = next(val_gen)
            x_val = tf.cast(x_val, tf.float32)
            preds = model(preprocess_fn(x_val), training=False)

            if y_val.ndim == 1 or (y_val.ndim == 2 and y_val.shape[1] == 1):
                val_loss = tf.keras.losses.sparse_categorical_crossentropy(y_val, preds)
            else:
                val_loss = tf.keras.losses.categorical_crossentropy(y_val, preds)

            val_losses.append(np.mean(val_loss.numpy()))
            val_pbar.set_postfix({"val_loss": f"{np.mean(val_losses):.4f}"})

        val_loss_mean = np.mean(val_losses) if len(val_losses) > 0 else float("inf")
        val_loss_history.append(val_loss_mean)
        print(f"Validation loss: {val_loss_mean:.4f}")

        # Early stopping + checkpoint 
        if epoch < 80:  # minimal training 80 epoch
            wait = 0
        elif val_loss_mean < best_val_loss:
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

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label='Training Loss (TRADES)')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title("TRADES Adversarial Training Loss — Brain MRI")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
