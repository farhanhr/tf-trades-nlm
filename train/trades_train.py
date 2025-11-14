import numpy as np
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tqdm import tqdm
from configs.train_config import config
from data.data_loader import get_training_data
from data.preprocessors import compose_preprocessors, nlm_denoise_bpda, bilateral_denoise_tf
from models.load_models import load_models
from models.trades_loss import trades_loss_tf, kl_div
from inference.attacks import generate_adversarial_examples_PGD

def trades_training_loop():
    epochs = config["epochs"]
    step_size = config["step_size"]
    epsilon = config["epsilon"]
    perturb_steps = config["perturb_steps"]
    beta = config["beta"]
    
    saved_models_path = config["saved_models_path"] + "BrainTumorMRI_VGG16_03112025_2358_model.h5"
    model = load_models(saved_models_path, compile=True, trainable=True)
    preprocess_fn = compose_preprocessors([
    bilateral_denoise_tf(spatial_sigma=1.0, intensity_sigma=25.0, kernel_size=3),
    nlm_denoise_bpda(h=0.3, patch_size=3, window_size=5),
    vgg_preprocess
    ])

    train_gen, val_gen = get_training_data(config["dataset_path"], 16, preprocessing_function=None)
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    max_lr = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=max_lr,
        first_decay_steps=steps_per_epoch*10,
        t_mul=2.0,               
        m_mul=0.9,              
        alpha=0.01           
    )
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, clipvalue=1.0, nesterov=False)

    best_val_loss, wait, patience = float("inf"), 0, 10

    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    method_name = f"TRADES_NLM_BF_B({beta})EPS({epsilon})LR({max_lr})"
    checkpoint_TRADES_model_dir = f"{config['checkpoint_dir']}{timestamp}/[Best_checkpoint]{method_name}_model_{timestamp}.h5"
    final_TRADES_model_dir = f"{config['final_model_dir']}[Final]{method_name}_model_{timestamp}.h5"

    train_loss_history, val_loss_history = [], []
    clean_acc_history, robust_acc_history = [], []
    kl_val_history, trades_val_history = [], []

    for epoch in range(epochs):
        current_lr = optimizer.learning_rate(optimizer.iterations).numpy()
        print(f"Epoch {epoch+1}/{epochs} — Current LR: {current_lr:.6f}")


        batch_losses, loss_nat_batch, loss_robust_batch = [], [], []
        train_pbar = tqdm(enumerate(train_gen), total=len(train_gen), desc="Training", ncols=100)
        for step, (x_batch_raw, y_batch) in train_pbar:
            x_batch_raw = tf.cast(x_batch_raw, tf.float32)

            loss, loss_nat, loss_robust = trades_loss_tf(
                model=model,
                x_natural=x_batch_raw,
                y=y_batch,
                optimizer=optimizer,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                beta=beta,
                preprocess_fn=preprocess_fn
            )

            batch_losses.append(float(loss.numpy()))
            loss_nat_batch.append(float(loss_nat.numpy()))
            loss_robust_batch.append(float(loss_robust.numpy()))
            if (step + 1) % 10 == 0:
                train_pbar.set_postfix({"loss": f"{np.mean(batch_losses[-10:]):.4f}"})

            if step + 1 >= len(train_gen):
                break

        train_loss = np.mean(batch_losses)
        nat_loss_mean = np.mean(loss_nat_batch)
        robust_loss_mean = np.mean(loss_robust_batch)
        train_loss_history.append(train_loss)
        print(f"Epoch {epoch+1} mean Trades loss: {train_loss:.4f}")
        print(f"Mean TRADES_nat_loss: {nat_loss_mean:.4f}")
        print(f"Mean TRADES_robust_loss: {(robust_loss_mean * beta):.4f}")

        #Validation Loss
        val_losses = []
        kl_divs = []
        val_pbar = tqdm(range(len(val_gen)), desc="Validation", ncols=100)
        clean_correct, adv_correct, total = 0, 0, 0
        for step in val_pbar:
            x_val, y_val = next(val_gen)
            x_val = tf.cast(x_val, tf.float32)
            y_val = tf.convert_to_tensor(y_val)
            preds_clean = model(preprocess_fn(x_val), training=False)
            val_loss = tf.keras.losses.sparse_categorical_crossentropy(y_val, preds_clean)
            val_losses.append(np.mean(val_loss.numpy()))
            val_pbar.set_postfix({"val_loss": f"{np.mean(val_losses):.4f}"})

            x_adv = generate_adversarial_examples_PGD(model, x_val, epsilon, step_size, perturb_steps)
            preds_adv = model(preprocess_fn(x_adv), training=False)
            
            # KL Divergence
            kl_diver = kl_div(preds_clean, preds_adv)
            kl_divs.append(kl_diver.numpy())
            
            clean_correct += tf.reduce_sum(tf.cast(tf.argmax(preds_clean, 1) == tf.cast(y_val, tf.int64), tf.float32)).numpy()
            adv_correct += tf.reduce_sum(tf.cast(tf.argmax(preds_adv, 1) == tf.cast(y_val, tf.int64), tf.float32)).numpy()
            total += y_val.shape[0]
            

        val_loss_mean = np.mean(val_losses) if len(val_losses) > 0 else float("inf")
        kl_mean = np.mean(kl_divs) if len(kl_divs) > 0 else float("inf")
        
        clean_acc = clean_correct / total
        robust_acc = adv_correct / total
        TRADES_val_loss_mean = val_loss_mean + beta * kl_mean
        
        val_loss_history.append(val_loss_mean)
        robust_acc_history.append(robust_acc)
        clean_acc_history.append(clean_acc)
        kl_val_history.append(kl_mean)
        trades_val_history.append(TRADES_val_loss_mean)
        print(f"Validation loss: {val_loss_mean:.4f}")
        print(f"KL val Mean: {kl_mean:.4f}")
        print(f"TRADES Validation loss mean: {TRADES_val_loss_mean:.4f}") 
        print(f"Clean Accuracy: {clean_acc:.6f} | Robust Accuracy: {robust_acc:.6f}")

        # Early Stopping + Checkpoint
        if (TRADES_val_loss_mean < best_val_loss) and epoch > 10:
            print(f"trades loss mean improved from {best_val_loss:.4f} → {TRADES_val_loss_mean:.4f}")
            best_val_loss = TRADES_val_loss_mean
            wait = 0
            model.save(checkpoint_TRADES_model_dir)
            print(f"Checkpoint saved at: {checkpoint_TRADES_model_dir}")
        elif epoch > 10:
            wait += 1
            print(f"No improvement from {best_val_loss:.4f}. Patience counter: {wait}/{patience}")

        if wait >= patience:
            print("Early stopping triggered — no improvement for several epochs.")
            break

    print(f"\nTraining done. Saved final model to: {final_TRADES_model_dir}")
    model.save(final_TRADES_model_dir)

    #Visualization & Save
    vis_dir = os.path.join(config["saved_models_path"], "visualization_data", f"{timestamp}_{method_name}")
    os.makedirs(vis_dir, exist_ok=True)

    # Dictionary metrik
    metrics = {
        "Trades_Loss": train_loss_history,
        "Clean_Validation_Loss": val_loss_history,
        "Kullback-Leibler Divergence at Validation": kl_val_history,
        "Trades_Validation_Loss": trades_val_history,
        "Clean_Accuracy": clean_acc_history,
        "Robust_Accuracy": robust_acc_history
    }
    for name, history in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(history, label=name)
        plt.title(f"{name} — TRADES Training")
        plt.xlabel("Epoch")
        plt.ylabel(name.replace("_", " "))
        plt.ylim(0, max(history) * 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        file_path = os.path.join(vis_dir, f"{name}_{timestamp}.png")
        plt.savefig(file_path)
        print(f"[INFO] {name} plot saved at: {file_path}")
        plt.close()
