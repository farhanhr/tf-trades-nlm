from sklearn.metrics import classification_report
import pandas as pd
import tensorflow as tf
from models.load_models import load_models
from configs.train_config import config

def model_classification_report(model_name, test_gen):
    model_path = config["saved_models_path"] + model_name
    model = load_models(model_path, compile=True)
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Model Evaluation: {model_name}")
    print(f"Test Accuracy: {test_acc:.2f}\n Test Loss: {test_loss:.2f} \n")

    y_pred_probs = model.predict(test_gen)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).T
    print(df_report)
