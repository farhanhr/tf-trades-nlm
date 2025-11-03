import tensorflow as tf
from typing import Callable, Optional

EPS = 1e-12

@tf.custom_gradient
def nlm_denoise_bpda_gpu_subsampled(
    x: tf.Tensor,
    h: float = 0.1,
    patch_size: int = 3,
    window_size: int = 7,
    step: int = 2  # subsampling stride, 1=full NLM, 2=faster
) -> tf.Tensor:
    assert patch_size % 2 == 1 and window_size % 2 == 1
    x_dtype = x.dtype
    x_f = tf.cast(x, tf.float32) / 255.0

    pad_r = (window_size - 1) // 2
    x_f = tf.pad(x_f, [[0,0],[pad_r,pad_r],[pad_r,pad_r],[0,0]], mode='REFLECT')

    B, H, W, C = tf.unstack(tf.shape(x_f))
    k = patch_size
    D = k * k * C

    patches = tf.image.extract_patches(
        images=x_f,
        sizes=[1, k, k, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )
    patches = tf.reshape(patches, [B, H, W, D])
    center = patches

    weighted_sum = tf.zeros_like(x_f)
    weight_sum = tf.zeros([B, H, W, 1], dtype=tf.float32)
    h_sq = tf.cast(h ** 2, tf.float32)

    # subsampled loop over neighborhood
    for dy in tf.range(-pad_r, pad_r + 1, step):
        for dx in tf.range(-pad_r, pad_r + 1, step):
            patches_shifted = tf.roll(patches, shift=[dy, dx], axis=[1, 2])
            img_shifted = tf.roll(x_f, shift=[dy, dx], axis=[1, 2])

            diff = center - patches_shifted
            dist2 = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)

            w = tf.exp(-dist2 / (h_sq + 1e-12))
            weighted_sum += w * img_shifted
            weight_sum += w

    denoised = weighted_sum / (weight_sum + 1e-12)
    denoised = denoised[:, pad_r:-pad_r, pad_r:-pad_r, :]
    denoised_scaled = tf.clip_by_value(denoised * 255.0, 0.0, 255.0)
    denoised_scaled = tf.cast(denoised_scaled, x_dtype)

    # GRAD: return single grad (upstream) because only `x` is tensor input in your calls
    def grad(upstream):
        return upstream

    return denoised_scaled, grad


def preprocess_with_nlm_gpu(
    x_raw: tf.Tensor,
    use_nlm: bool = True,
    h: float = 0.1,
    patch_size: int = 3,
    window_size: int = 7,
    step: int = 2,  #subsampling
    vgg_preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tf.Tensor:
    x = tf.cast(x_raw, tf.float32)
    if use_nlm:
        x = nlm_denoise_bpda_gpu_subsampled(
            x, h=h, patch_size=patch_size,
            window_size=window_size, step=step
        )
    if vgg_preprocess_fn is not None:
        x = vgg_preprocess_fn(x)
    return x
