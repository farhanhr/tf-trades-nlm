import os
import tensorflow as tf
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from inference.evaluate_ART import Tee, _ensure_list
from models.load_models import load_models
from configs.train_config import config
import os
import sys
from datetime import datetime
from contextlib import redirect_stdout
from inference.inference_helper import save_confusion_matrix

def attack_mim_cleverhans(model, x, eps, step_size, nb_iter=10, clip_values=(0,255)):
    """MIM attack CleverHans — L∞."""
    x_adv = momentum_iterative_method(
        model_fn=model,
        x=x,
        eps=eps,
        eps_iter=step_size,
        nb_iter=nb_iter,
        clip_min=clip_values[0],
        clip_max=clip_values[1]
    )
    return x_adv


def attack_rfgsm(model, x, y_true, eps, clip_values=(0, 255), from_logits=False):
    x = tf.cast(x, tf.float32)

    noise = tf.random.uniform(tf.shape(x), -eps, eps)
    x_init = x + noise

    x_init = tf.clip_by_value(x_init, clip_values[0], clip_values[1])

    with tf.GradientTape() as tape:
        tape.watch(x_init)
        preds = model(x_init, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true, preds, from_logits=from_logits
            )
        )

    grad = tape.gradient(loss, x_init)
    grad_sign = tf.sign(grad)

    x_adv = x_init + eps * grad_sign

    x_adv = x + tf.clip_by_value(x_adv - x, -eps, eps)

    x_adv = tf.clip_by_value(x_adv, clip_values[0], clip_values[1])

    return x_adv


  
def batch_evaluation_cleverhans_attacks(
    model_name,
    test_gen,
    epsilon=2.55,
    step_size=0.51,
    preprocess_fn=None,
    batch_size_eval=16,
    nb_classes=4,
    input_shape=(224,224,3),
    clip_values=(0.0,255.0),
    out_dir=".",
    timestamp_fmt="%Y%m%d_%H%M%S",
):
    """
    Evaluasi banyak model + epsilon memakai CleverHans attacks:
       - MIM
       - RFGSM
       - MI-FAB (approx)
    """
    models = _ensure_list(model_name)
    eps_list = _ensure_list(epsilon)
    step_list = _ensure_list(step_size)
    preprocess_list = _ensure_list(preprocess_fn) if preprocess_fn else [None]

    if len(preprocess_list) == 1:
        preprocess_list = preprocess_list * len(models)

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime(timestamp_fmt)
    filename = os.path.join(out_dir, f"{timestamp}_Evaluasi_CleverHans.txt")

    with open(filename, "w") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print("="*80)
            print("   EVALUASI CLEVERHANS ATTACKS")
            print("="*80)
            print("Models:", models)
            print("Epsilons:", eps_list)
            print("Step Sizes:", step_list)

            x_list, y_list = [], []
            for i in range(len(test_gen)):
                xb, yb = test_gen[i]
                x_list.append(xb.astype(np.float32))
                y_list.append(yb.astype(np.int32))

            x_all = np.concatenate(x_list, axis=0)
            y_all = np.concatenate(y_list, axis=0)
            target_names = list(test_gen.class_indices.keys())

            for m_idx, (mpath, prep_fn) in enumerate(zip(models, preprocess_list)):
                print("\n" + "="*5, f"Model ({m_idx+1}/{len(models)}) {mpath}", "="*5)

                base_model = load_models(config["saved_models_path"] + mpath, compile=True)

                for eps, step in zip(eps_list, step_list):

                    print("\n---- Evaluasi Epsilon =", eps, "-----")

                    attacks = {
                        "MIM": lambda m,x: attack_mim_cleverhans(m, x, eps, step),
                        "RFGSM": lambda m, x, y: attack_rfgsm(m, x, y, eps)
                    }

                    for atk_name, atk_fn in attacks.items():
                        print(f"\n=== Serangan {atk_name} epsilon=({eps}) ===")

                        preds_adv_all = []
                        for i in range(0, len(x_all), batch_size_eval):
                            xb = x_all[i:i+batch_size_eval]
                            yb = y_all[i:i+batch_size_eval]
                            xb_tf = tf.convert_to_tensor(xb, dtype=tf.float32)

                            # generate attack
                            if atk_name == "RFGSM":
                              x_adv = atk_fn(base_model, xb_tf, yb)
                            else:
                              x_adv = atk_fn(base_model, xb_tf)
                              
                            x_adv = x_adv.numpy()

                            if prep_fn:
                                x_pre = prep_fn(tf.convert_to_tensor(x_adv)).numpy()
                            else:
                                x_pre = x_adv

                            preds_adv_all.append(base_model.predict(x_pre, verbose=0))

                        preds_adv = np.concatenate(preds_adv_all, axis=0)
                        y_pred = preds_adv.argmax(axis=1)
                        acc = (y_pred == y_all).mean()

                        print(f"Akurasi terhadap {atk_name}: {acc:.2%}")
                        print(classification_report(y_all, y_pred, target_names=target_names))
                        save_confusion_matrix(
                            y_true=y_all,
                            y_pred=y_pred,
                            target_names=target_names,
                            out_dir=out_dir,
                            model_name=mpath,
                            attack_name=atk_name,
                            epsilon=eps,
                            timestamp=timestamp,
                            cmap="Reds"
                        )

    return filename
