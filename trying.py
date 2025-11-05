from configs.train_config import config
from data.preprocessors import preprocess_with_nlm_gpu, preprocess_median_gaussian
from data.data_loader import get_testing_data, get_adversarial_data
import matplotlib.pyplot as plt
import numpy as np


# preprocess_fn = lambda x: preprocess_with_nlm_gpu(
#     x,
#     use_nlm=True,
#     h=0.15,
#     patch_size=3,
#     window_size=5,
# )
preprocess_fn = lambda x: preprocess_median_gaussian(
    x,
    use_preproc=True,
    median_size=3,
    gaussian_sigma=0.8,
    gaussian_kernel=5,
    # vgg_preprocess_fn=vgg_preprocess
)
train_gen = get_adversarial_data(config["dataset_path"], 2)
# train_gen, _ = get_training_data(config["dataset_path"], 6)
    
batch_images, _ = next(train_gen)

num_samples = min(5, len(batch_images))
sample_images = batch_images[:num_samples]

preprocessed_images = []
for img in sample_images:
    img_in = np.expand_dims(img, axis=0)  # tambahkan batch dimension -> (1, H, W, C)
    preprocessed = preprocess_fn(img_in)
    
    # kalau output-nya tetap batch (1, H, W, C), buang dimensi batch
    if hasattr(preprocessed, "numpy"):
        preprocessed = preprocessed.numpy()
    preprocessed = np.squeeze(preprocessed, axis=0)
    
    preprocessed_images.append(preprocessed)

preprocessed_images = np.array(preprocessed_images)


fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3 * num_samples))

for i in range(num_samples):
    before = sample_images[i]
    after = preprocessed_images[i]

    if before.shape[0] in [1, 3]:
        before = np.moveaxis(before, 0, -1)
    if after.shape[0] in [1, 3]:
        after = np.moveaxis(after, 0, -1)

    before = np.clip(before / 255.0, 0, 1)
    after = np.clip(after / 255.0, 0, 1)

    axes[i, 0].imshow(before)
    axes[i, 0].set_title("Before Preprocessing")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(after)
    axes[i, 1].set_title("After Preprocessing")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

