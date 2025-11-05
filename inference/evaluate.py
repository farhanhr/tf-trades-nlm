from data.data_loader import get_testing_data
from models.load_models import load_models
from configs.train_config import config

test_gen = get_testing_data(config["dataset_path"], config["batch_size"])
saved_models_path = config["saved_models_path"] + "BrainTumorMRI_VGG16_26102025_0850_model.h5"
model = load_models(saved_models_path, compile=False)

test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.2f}")