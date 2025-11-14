import tensorflow as tf
import numpy as np
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import (
    FastGradientMethod, 
    ProjectedGradientDescent, 
    DeepFool, 
    AutoProjectedGradientDescent,
    SquareAttack
)
from sklearn.metrics import classification_report
from models.load_models import load_models
from configs.train_config import config
import os
import sys
from datetime import datetime
from contextlib import redirect_stdout
from inference.inference_helper import make_logits_model_if_softmax, save_confusion_matrix



def evaluate(model_name, test_gen, epsilon, step_size=0.51, preprocess_fn=None, batch_size_eval=16):
    """
    Evaluasi model terhadap adversarial attacks (FGSM, PGD)
    Serangan dilakukan di domain [0,255], kemudian input dipreproses sebelum inferensi.
    """
    model_path = config["saved_models_path"] + model_name
    base_model = load_models(model_path, compile=True)

    classifier = TensorFlowV2Classifier(
        model=base_model,
        nb_classes=4,
        input_shape=(224, 224, 3),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        clip_values=(0.0, 255.0),
    )

    attacks = {
        "FGSM": FastGradientMethod(estimator=classifier, eps=epsilon),
        "PGD": ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=step_size, max_iter=10),
        "DeepFool": DeepFool(classifier=classifier, epsilon=epsilon, max_iter=50),
    }

    x_list, y_list = [], []
    for i in range(len(test_gen)):
        xb, yb = test_gen[i]
        x_list.append(xb)
        y_list.append(yb)
    x_all = np.concatenate(x_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int32)

    print("\n=== Evaluasi Clean ===")
    preds_clean = []
    for i in range(0, len(x_all), batch_size_eval):
        xb = x_all[i:i + batch_size_eval]

        if preprocess_fn is not None:
            xb_pre = preprocess_fn(tf.convert_to_tensor(xb))
        else:
            xb_pre = xb

        preds = base_model.predict(xb_pre, batch_size=batch_size_eval, verbose=0)
        preds_clean.append(preds)

    preds_clean = np.concatenate(preds_clean, axis=0)
    acc_clean = (preds_clean.argmax(axis=1) == y_all).mean()
    print(f"Akurasi Natural (tanpa serangan): {acc_clean:.2%}")

    for name, atk in attacks.items():
        print(f"\n=== Menjalankan serangan {name} (ε={epsilon}) ===")
        preds_adv_all = []
        # linf_vals = []

        for i in range(0, len(x_all), batch_size_eval):
            xb = x_all[i:i + batch_size_eval]
            x_adv = atk.generate(x=xb)


            if preprocess_fn is not None:
                x_adv_pre = preprocess_fn(tf.convert_to_tensor(x_adv))
            else:
                x_adv_pre = x_adv

            preds_adv = base_model.predict(x_adv_pre, batch_size=batch_size_eval, verbose=0)
            preds_adv_all.append(preds_adv)

        preds_adv = np.concatenate(preds_adv_all, axis=0)
        acc_adv = (preds_adv.argmax(axis=1) == y_all).mean()
        print(f"Akurasi terhadap {name}: {acc_adv:.2%}")

    tf.keras.backend.clear_session()

def classification_report_art(model_name, test_gen, epsilon=2.55, step_size=0.51, preprocess_fn=None, batch_size_eval=16):
    """
    Evaluasi model terhadap adversarial attacks (FGSM & PGD) 
    dan menampilkan classification report seperti sklearn.
    """
    model_path = config["saved_models_path"] + model_name
    base_model = load_models(model_path, compile=True)

    classifier = TensorFlowV2Classifier(
        model=base_model,
        nb_classes=4,
        input_shape=(224, 224, 3),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        clip_values=(0.0, 255.0),
    )
    logits_model = make_logits_model_if_softmax(base_model)

    if logits_model is not None:
        print("Model mengandung softmax di akhir -> menggunakan dual classifier (logits & probs).")
        model_logits = logits_model
    else:
        print("Tidak menemukan softmax eksplisit -> menggunakan model sama untuk logits & probs classifier.")
        model_logits = base_model

    classifier_logits = TensorFlowV2Classifier(
        model=model_logits,
        nb_classes=4,
        input_shape=(224, 224, 3),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        clip_values=(0.0, 255.0),
    )

    x_list, y_list = [], []
    for i in range(len(test_gen)):
        xb, yb = test_gen[i]
        x_list.append(xb)
        y_list.append(yb)
    x_all = np.concatenate(x_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int32)

    print("\n=== Evaluasi Clean ===")
    preds_clean = []
    for i in range(0, len(x_all), batch_size_eval):
        xb = x_all[i:i + batch_size_eval]
        xb_pre = preprocess_fn(tf.convert_to_tensor(xb)) if preprocess_fn else xb
        preds_clean.append(base_model.predict(xb_pre, batch_size=batch_size_eval, verbose=0))

    preds_clean = np.concatenate(preds_clean, axis=0)
    y_pred_clean = preds_clean.argmax(axis=1)
    acc_clean = (y_pred_clean == y_all).mean()
    target_names = list(test_gen.class_indices.keys())
    print(f"Akurasi Natural (tanpa serangan): {acc_clean:.2%}")
    print("\nClassification Report (Clean):")
    print(classification_report(y_all, y_pred_clean, target_names=target_names, digits=4))
    H, W, C = 224, 224, 3
    eps_l2 = np.sqrt(H * W * C) * epsilon
    attacks = {
        "FGSM": FastGradientMethod(estimator=classifier, eps=epsilon),
        "PGD_Linf": ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=step_size, max_iter=10),
        "APGD_Linf": AutoProjectedGradientDescent(estimator=classifier_logits, eps=epsilon, eps_step=step_size, loss_type="cross_entropy", verbose=False),
        "PGD_L2": ProjectedGradientDescent(estimator=classifier, norm=2, eps=eps_l2, eps_step=(eps_l2/10), max_iter=10),
        "APGD_L2": AutoProjectedGradientDescent(estimator=classifier_logits, norm=2, eps=eps_l2, eps_step=(eps_l2/10), loss_type="cross_entropy", verbose=False),
        "DeepFool": DeepFool(classifier=classifier, verbose=False),
    }

    for atk_name, atk in attacks.items():
        print(f"\n=== Menjalankan serangan {atk_name} (ε={epsilon}) ===")
        preds_adv_all = []
        for i in range(0, len(x_all), batch_size_eval):
            xb = x_all[i:i + batch_size_eval]
            x_adv = atk.generate(x=xb)
            x_adv_pre = preprocess_fn(tf.convert_to_tensor(x_adv)) if preprocess_fn else x_adv
            preds_adv_all.append(base_model.predict(x_adv_pre, batch_size=batch_size_eval, verbose=0))

        preds_adv = np.concatenate(preds_adv_all, axis=0)
        y_pred_adv = preds_adv.argmax(axis=1)
        acc_adv = (y_pred_adv == y_all).mean()
        target_names = list(test_gen.class_indices.keys())
        print(f"Akurasi terhadap {atk_name}: {acc_adv:.2%}")
        print(f"\nClassification Report ({atk_name}):")
        print(classification_report(y_all, y_pred_adv, target_names=target_names, digits=4))

    tf.keras.backend.clear_session()
    

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

def _ensure_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    else:
        return [x]

def batch_evaluation(
    model_name,
    test_gen,
    epsilon=2.55,
    step_size=0.51,
    preprocess_fn=None,
    batch_size_eval=16,
    nb_classes=4,
    input_shape=(224,224,3),
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    clip_values=(0.0, 255.0),
    out_dir=".",
    timestamp_fmt="%Y%m%d_%H%M%S",
    max_runs=1000
):
    """
    Versi baru (mandiri) yang menjalankan evaluasi adversarial untuk banyak model dan epsilon,
    menyimpan semua output ke file teks <timestamp>_Evaluasi_ART.txt.

    Perubahan utama: preprocess_fn boleh berupa:
      - satu fungsi (dipakai untuk semua model)
      - list of functions dengan panjang == jumlah model (function ke-i dipakai untuk model ke-i)

    Argumen:
      - model_name: str atau list of str (relatif ke config['saved_models_path'])
      - preprocess_fn: None, callable, atau list of callables aligned with `model_name`
      - epsilon, step_size: float atau list
      - max_runs: batas max kombinasi model*epsilon (None untuk non-cek)
    """

    models = _ensure_list(model_name)
    eps_list = _ensure_list(epsilon)
    step_list = _ensure_list(step_size)
    preprocess_list = _ensure_list(preprocess_fn) if preprocess_fn is not None else [None]


    if len(models) == 0:
        raise ValueError("model_name tidak boleh kosong.")
    if len(eps_list) == 0:
        raise ValueError("epsilon tidak boleh kosong.")

    if len(step_list) == 1 and len(eps_list) > 1:
        step_list = step_list * len(eps_list)
    elif len(step_list) != len(eps_list):
        step_list = list(np.resize(step_list, len(eps_list)))

    if len(preprocess_list) == 1:
        preprocess_list = preprocess_list * len(models)
    elif len(preprocess_list) != len(models):
        raise ValueError(
            f"Jika preprocess_fn diberikan sebagai list, panjangnya harus sama dengan jumlah model. "
            f"Jumlah models={len(models)}, panjang preprocess_fn list={len(preprocess_list)}"
        )

    total_runs = len(models) * len(eps_list)
    if max_runs is not None and total_runs > max_runs:
        raise ValueError(
            f"Jumlah kombinasi (model*epsilon) = {total_runs} melebihi batas max_runs={max_runs}. "
            "Kurangi jumlah model/epsilon atau naikkan max_runs jika benar-benar ingin menjalankan semuanya."
        )

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime(timestamp_fmt)
    filename = os.path.join(out_dir, f"{timestamp}_Evaluasi_ART.txt")

    with open(filename, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print("="*80)
            print(f"WAKTU MULAI: {datetime.now().isoformat()}")
            print(f"MODELS: {models}")
            print(f"EPSILONS: {eps_list}")
            print(f"STEP_SIZES: {step_list}")
            print(f"PREPROCESS_FUNCS_PER_MODEL: {[None if p is None else getattr(p,'__name__',str(p)) for p in preprocess_list]}")
            print("="*80)

            for m_idx, (mpath, prep_fn) in enumerate(zip(models, preprocess_list)):
                print("\n" + "="*5 + f" Mengevaluasi Model ({m_idx+1}/{len(models)}): {mpath} " + "="*5)
                model_path = config["saved_models_path"] + mpath
                try:
                    base_model = load_models(model_path, compile=True)
                except Exception as e:
                    print(f"[ERROR] Gagal load model {mpath}: {repr(e)}")

                    continue

                try:
                    classifier = TensorFlowV2Classifier(
                        model=base_model,
                        nb_classes=nb_classes,
                        input_shape=input_shape,
                        loss_object=loss_object,
                        clip_values=clip_values,
                    )
                    logits_model = make_logits_model_if_softmax(base_model)

                    if logits_model is not None:
                        print("Model mengandung softmax di akhir -> menggunakan dual classifier (logits & probs).")
                        model_logits = logits_model
                    else:
                        print("Tidak menemukan softmax eksplisit -> menggunakan model sama untuk logits & probs classifier.")
                        model_logits = base_model

                    classifier_logits = TensorFlowV2Classifier(
                        model=model_logits,
                        nb_classes=4,
                        input_shape=(224, 224, 3),
                        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        clip_values=(0.0, 255.0),
                    )
                except Exception as e:
                    print(f"[ERROR] Gagal membuat ART classifier untuk {mpath}: {repr(e)}")
                    tf.keras.backend.clear_session()
                    continue

                x_list, y_list = [], []
                try:
                    for i in range(len(test_gen)):
                        xb, yb = test_gen[i]
                        x_list.append(xb)
                        y_list.append(yb)
                except Exception as e:
                    print(f"[ERROR] Gagal membaca test_gen: {repr(e)} - pastikan test_gen mendukung indexing seperti sebelumnya")
                    tf.keras.backend.clear_session()
                    continue

                x_all = np.concatenate(x_list, axis=0).astype(np.float32)
                y_all = np.concatenate(y_list, axis=0).astype(np.int32)
                target_names = list(test_gen.class_indices.keys())

                print("\n=== Evaluasi Clean ===")
                preds_clean = []
                for i in range(0, len(x_all), batch_size_eval):
                    xb = x_all[i:i + batch_size_eval]
                    xb_pre = prep_fn(tf.convert_to_tensor(xb)) if prep_fn else xb
                    preds_clean.append(base_model.predict(xb_pre, batch_size=batch_size_eval, verbose=0))
                preds_clean = np.concatenate(preds_clean, axis=0)
                y_pred_clean = preds_clean.argmax(axis=1)
                acc_clean = (y_pred_clean == y_all).mean()
                print(f"Akurasi Natural (tanpa serangan): {acc_clean:.2%}")
                print("\nClassification Report (Clean):")
                print(classification_report(y_all, y_pred_clean, target_names=target_names, digits=4))
                save_confusion_matrix(
                    y_true=y_all,
                    y_pred=y_pred_clean,
                    target_names=target_names,
                    out_dir=out_dir,
                    model_name=mpath,
                    attack_name="Clean",
                    timestamp=timestamp,
                    cmap="Blues"
                )

                for eps_idx, (eps, step) in enumerate(zip(eps_list, step_list)):
                    print("\n" + "-"*6 + f" Kombinasi (eps {eps_idx+1}/{len(eps_list)}): epsilon={eps}, step_size={step} " + "-"*6)

                    H, W, C = input_shape
                    try:
                        eps_float = float(eps)
                        step_float = float(step)
                    except Exception:
                        print(f"[ERROR] epsilon atau step_size tidak konversi ke float: eps={eps}, step={step}")
                        continue
                    eps_l2 = np.sqrt(H * W * C) * eps_float

                    try:
                        attacks = {
                            "FGSM": FastGradientMethod(estimator=classifier, eps=eps_float),
                            "PGD_Linf": ProjectedGradientDescent(estimator=classifier, eps=eps_float, eps_step=step_float, max_iter=10,),
                            # "APGD_Linf": AutoProjectedGradientDescent(estimator=classifier_logits, eps=eps_float, eps_step=step_float, loss_type='difference_logits_ratio', verbose=False),
                            "PGD_L2": ProjectedGradientDescent(estimator=classifier, norm=2, eps=eps_l2, eps_step=(eps_l2/10), max_iter=10),
                            # "Square": SquareAttack(estimator=classifier, eps=eps_float, max_iter=100, verbose=False)
                            # "APGD_L2": AutoProjectedGradientDescent(estimator=classifier_logits, norm=2, eps=eps_l2, eps_step=(eps_l2/10), loss_type='difference_logits_ratio', verbose=False),
                            # "DeepFool": DeepFool(classifier=classifier, verbose=False),
                        }
                    except Exception as e:
                        print(f"[ERROR] Gagal inisialisasi attacks untuk eps={eps}: {repr(e)}")
                        continue

                    for atk_name, atk in attacks.items():
                        print(f"\n=== Menjalankan serangan {atk_name} (ε={eps}) ===")
                        preds_adv_all = []
                        try:
                            for i in range(0, len(x_all), batch_size_eval):
                                xb = x_all[i:i + batch_size_eval]
                                
                                x_adv = atk.generate(x=xb)
                                x_adv_pre = prep_fn(tf.convert_to_tensor(x_adv)) if prep_fn else x_adv
                                preds_adv_all.append(base_model.predict(x_adv_pre, batch_size=batch_size_eval, verbose=0))
                        except Exception as e:
                            print(f"[ERROR] Saat generate/predict adversarial ({atk_name}) untuk model={mpath}, eps={eps}: {repr(e)}")
                            
                            continue

                        try:
                            preds_adv = np.concatenate(preds_adv_all, axis=0)
                            y_pred_adv = preds_adv.argmax(axis=1)
                            acc_adv = (y_pred_adv == y_all).mean()
                            print(f"Akurasi terhadap {atk_name}: {acc_adv:.2%}")
                            print(f"\nClassification Report ({atk_name}):")
                            print(classification_report(y_all, y_pred_adv, target_names=target_names, digits=4))
                            save_confusion_matrix(
                                y_true=y_all,
                                y_pred=y_pred_adv,
                                target_names=target_names,
                                out_dir=out_dir,
                                model_name=mpath,
                                attack_name=atk_name,
                                epsilon=eps,
                                timestamp=timestamp,
                                cmap="Reds"
                            )

                        except Exception as e:
                            print(f"[ERROR] Saat menghitung metric/pelaporan untuk attack {atk_name}: {repr(e)}")
                            continue

                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass

            print("\n" + "="*40)
            print(f"WAKTU SELESAI: {datetime.now().isoformat()}")
            print(f"Hasil disimpan di: {filename}")
            print("="*40)

    return filename