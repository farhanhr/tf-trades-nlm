import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def make_logits_model_if_softmax(base_model):
    """
    Jika base_model berakhir dengan softmax (baik sebagai activation pada layer terakhir
    atau Activation layer terpisah), kembalikan model baru yang output-nya pre-softmax (logits).
    Jika tidak bisa, kembalikan None.
    """
    try:
        last = base_model.layers[-1]
    except Exception:
        return None

    if hasattr(last, 'activation') and last.activation == tf.keras.activations.softmax:
        try:
            preact = last.input  
            logits_model = tf.keras.Model(inputs=base_model.input, outputs=preact)
            return logits_model
        except Exception:
            return None

    if isinstance(last, Activation) and last.activation == tf.keras.activations.softmax:
        try:
            preact = base_model.layers[-2].output
            logits_model = tf.keras.Model(inputs=base_model.input, outputs=preact)
            return logits_model
        except Exception:
            return None

    return None


def save_confusion_matrix(y_true, y_pred, target_names, out_dir, 
                          model_name="", attack_name="Clean", epsilon=None, timestamp=None,
                          cmap="Blues"):
    """
    Membuat dan menyimpan confusion matrix (teks + gambar).
    
    Argumen:
      - y_true, y_pred : label sebenarnya dan prediksi model
      - target_names : list nama kelas (sesuai test_gen.class_indices.keys())
      - out_dir : folder tempat menyimpan file hasil (.png)
      - model_name : nama model (string)
      - attack_name : nama serangan ("Clean", "FGSM", "PGD", dst)
      - epsilon : nilai epsilon (opsional, ditampilkan di judul)
      - timestamp : string waktu unik (misal dari datetime.now().strftime(...))
      - cmap : colormap seaborn
    """

    # Buat confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Cetak ke log (agar muncul juga di file .txt hasil evaluasi)
    print(f"\nConfusion Matrix ({attack_name}):")
    print(cm)

    # Buat plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=target_names, yticklabels=target_names)
    
    title = f"Confusion Matrix - {attack_name}"
    if epsilon is not None:
        title += f" (ε={epsilon})"
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Buat nama file hasil
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    eps_str = f"_eps{epsilon}" if epsilon is not None else ""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_dir = os.path.join(out_dir, "confussion_matrix")
    os.makedirs(cm_dir, exist_ok=True)
    filename = os.path.join(cm_dir, f"{timestamp}_{safe_model}_{attack_name}{eps_str}_CM.png")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Confusion matrix plot disimpan di: {filename}")
    return filename