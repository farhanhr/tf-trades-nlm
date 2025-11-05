from data.data_loader import get_testing_data
from models.load_models import load_models
from configs.train_config import config
from sklearn.metrics import classification_report
import pandas as pd

test_gen = get_testing_data(config["dataset_path"], config["batch_size"])
saved_models_path = config["saved_models_path"] + "BrainTumorMRI_VGG16_03112025_2358_model.h5"
model = load_models(saved_models_path, compile=True)

test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.2f}")

y_pred_probs = model.predict(test_gen)
y_pred = y_pred_probs.argmax(axis=1)

y_true = test_gen.classes

class_names = list(test_gen.class_indices.keys())


report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).T

print(df_report)
