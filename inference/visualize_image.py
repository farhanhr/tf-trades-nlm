import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from inference.attacks import generate_adversarial_examples_PGD
from configs.train_config import config
from data.data_loader import get_testing_data

def normalize_for_display(x):
    """Normalize tensor to [0,1] for display purposes."""
    x = x - tf.reduce_min(x)
    x = x / (tf.reduce_max(x) + 1e-8)
    return x

def visualize_preprocessing_and_attack(model, preprocess_fn, epsilon=8, step_size=2, steps=10, max_samples=2):
    """
    Visualize multiple examples (up to `max_samples`) from different classes,
    including original, preprocessed, adversarial, and adversarial+preprocessed.
    """
    test_gen = get_testing_data(config["dataset_path"], batch_size=64, preprocessing_fn=None)
    x_all, y_all = next(iter(test_gen))
    x_all = tf.cast(x_all, tf.float32)
    y_all = tf.cast(y_all, tf.int32)

    unique_classes = tf.unique(y_all)[0].numpy()
    selected_indices = []

    for cls in unique_classes:
        cls_idx = np.where(y_all.numpy() == cls)[0]
        if len(cls_idx) > 0:
            selected_indices.append(cls_idx[0])
        if len(selected_indices) >= max_samples:
            break

    if len(selected_indices) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y_all)), selected_indices)
        random_pick = np.random.choice(remaining, size=max_samples - len(selected_indices), replace=False)
        selected_indices = np.concatenate([selected_indices, random_pick])

    x_sel = tf.gather(x_all, selected_indices)
    y_sel = tf.gather(y_all, selected_indices)
    num_samples = x_sel.shape[0]

    x_pre = preprocess_fn(x_sel)
    x_adv = generate_adversarial_examples_PGD(model, x_sel, epsilon, step_size, steps)
    x_adv = tf.cast(x_adv, tf.float32)
    x_adv_pre = preprocess_fn(x_adv)

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3 * num_samples))
    titles = ["Original", "Preprocessed", "Adversarial (PGD)", "Adv + Preprocessed"]

    for i in range(num_samples):
        imgs = [
            normalize_for_display(x_sel[i]),
            normalize_for_display(x_pre[i]),
            normalize_for_display(x_adv[i]),
            normalize_for_display(x_adv_pre[i]),
        ]
        for j in range(4):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(imgs[j])
            if j == 0:
                ax.set_ylabel(f"Class {y_sel[i].numpy()}", fontsize=12)
            ax.set_title(titles[j])
            ax.axis("off")

    plt.tight_layout()
    plt.show()
