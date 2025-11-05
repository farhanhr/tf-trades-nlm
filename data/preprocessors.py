import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from typing import Callable, Optional
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

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



@tf.function
def fast_median_filter(x: tf.Tensor, ksize: int = 3) -> tf.Tensor:
    """
    GPU-friendly median filter implemented with extract_patches.
    Input x: [B,H,W,C], float32 or float16. Returns [B,H,W,C].
    """
    # ensure float32 for stable sort ops
    x_f = tf.cast(x, tf.float32)
    B = tf.shape(x_f)[0]
    H = tf.shape(x_f)[1]
    W = tf.shape(x_f)[2]
    C = tf.shape(x_f)[3]

    patches = tf.image.extract_patches(
        images=x_f,
        sizes=[1, ksize, ksize, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'  # maintain spatial dims
    )
    # patches shape: [B, H, W, k*k*C]
    ksq = ksize * ksize

    patches_reshaped = tf.reshape(patches, [B, H, W, ksq, C])
    # sort along patch-window axis (3)
    patches_sorted = tf.sort(patches_reshaped, axis=3)
    median_index = ksq // 2
    median = patches_sorted[:, :, :, median_index, :]  # shape [B,H,W,C]
    # cast back to original dtype
    median = tf.cast(median, x.dtype)
    return median


@tf.function
def fast_gaussian_blur(x: tf.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> tf.Tensor:
    """
    Gaussian blur via separable kernel implemented with depthwise_conv2d for GPU.
    Input: [B,H,W,C]. Output same shape.
    """
    # Build separable gaussian kernel (float32)
    size = int(kernel_size)
    coords = tf.cast(tf.range(size), tf.float32) - (size - 1) / 2.0
    g1 = tf.exp(-(coords ** 2) / (2.0 * (tf.cast(sigma, tf.float32) ** 2)))
    g1 = g1 / tf.reduce_sum(g1)  # normalize

    # outer product to get 2D kernel
    kernel2d = tf.tensordot(g1, g1, axes=0)  # shape [k,k]
    kernel2d = tf.reshape(kernel2d, [size, size, 1, 1])  # [k,k,1,1]

    in_channels = tf.shape(x)[-1]
    # tile kernel across in_channels to shape [k,k,in_channels,1] (depthwise conv)
    kernel_depthwise = tf.tile(kernel2d, [1, 1, in_channels, 1])  

    x_f = tf.cast(x, tf.float32)
    blurred = tf.nn.depthwise_conv2d(x_f, kernel_depthwise, strides=[1, 1, 1, 1], padding='SAME')
    blurred = tf.cast(blurred, x.dtype)
    return blurred


@tf.function
def denoise_median_gaussian_gpu(x: tf.Tensor, median_ksize: int = 3, gaussian_kernel: int = 3, gaussian_sigma: float = 1.0) -> tf.Tensor:
    """
    Full denoising pipeline (median then gaussian) — GPU friendly.
    """
    x = tf.cast(x, tf.float32)
    x = fast_median_filter(x, ksize=median_ksize)
    x = fast_gaussian_blur(x, kernel_size=gaussian_kernel, sigma=gaussian_sigma)
    return tf.cast(x, tf.float32)  


def preprocess_median_gaussian_vgg(
    x: tf.Tensor,
    use_denoise: bool = True,
    median_size: int = 3,
    gaussian_kernel: int = 3,
    gaussian_sigma: float = 1.0
) -> tf.Tensor:
    """
    Robust preprocessing pipeline:
      - ensures input rank 4
      - converts grayscale->RGB when needed (graph-safe)
      - applies GPU-friendly median+gaussian denoising
      - applies VGG16 preprocess_input (BGR mean subtraction)
    Returns tensor ready for VGG model.
    """
    x = tf.cast(x, tf.float32)

    if tf.rank(x) == 3:
        x = tf.expand_dims(x, axis=0)  # [1,H,W,C]


    def _to_rgb():
        return tf.concat([x, x, x], axis=-1)

    def _keep():
        return x

    channel_count = tf.shape(x)[-1]
    x = tf.cond(tf.equal(channel_count, 1), _to_rgb, _keep)  

    if use_denoise:

        x = denoise_median_gaussian_gpu(x, median_ksize=median_size, gaussian_kernel=gaussian_kernel, gaussian_sigma=gaussian_sigma)

    x = vgg_preprocess(x)

    return x


def apply_preprocessing(gen, preprocess_fucntion):
    for batch_x, batch_y in gen:
        yield preprocess_fucntion(batch_x), batch_y
