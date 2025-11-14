import tensorflow as tf
import tensorflow_addons as tfa
from typing import Callable, List
EPS = 1e-12

def compose_preprocessors(steps: List[Callable[[tf.Tensor], tf.Tensor]]) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Compose multiple preprocessing functions into a single callable.
    Example:
        preprocess_fn = compose_preprocessors([
            bilateral_denoise_tf(spatial_sigma=2.0),
            total_variation_denoise_tf(weight=0.1),
            nlm_denoise_bpda(h=0.15),
            vgg_preprocess
        ])
    """
    def composed_fn(x: tf.Tensor) -> tf.Tensor:
        for fn in steps:
            x = fn(x)
        return x
    return composed_fn

#  Bilateral Denoiser (Tomasi & Manduchi, ICCV 1998)
def bilateral_denoise_tf(spatial_sigma=2.0, intensity_sigma=25.0, kernel_size=5) -> Callable[[tf.Tensor], tf.Tensor]:
    """Create a bilateral filter callable.""" 
    def _fn(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        x_blur = tfa.image.gaussian_filter2d(
            x, filter_shape=(kernel_size, kernel_size), sigma=spatial_sigma
        )
        diff = x - x_blur
        weight = tf.exp(-tf.square(diff) / (2.0 * (intensity_sigma ** 2)))
        x_denoised = (x_blur * weight + x * (1 - weight))
        return tf.clip_by_value(x_denoised, 0.0, 255.0)
    _fn.__name__ = "bilateral_denoise_tf"
    return _fn

# Total Variation Denoising (Rudin–Osher–Fatemi, 1992)
def total_variation_denoise_tf(weight=0.1, iterations=1) -> Callable[[tf.Tensor], tf.Tensor]:
    """Stable Total Variation denoising (Rudin–Osher–Fatemi, 1992)."""
    def _fn(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)

        for _ in range(iterations):
            # Compute spatial gradients with tf.roll to maintain consistent shapes
            dx = x - tf.roll(x, shift=1, axis=2)  # Horizontal difference
            dy = x - tf.roll(x, shift=1, axis=1)  # Vertical difference

            # Compute isotropic gradient magnitude
            grad_magnitude = tf.sqrt(dx**2 + dy**2 + 1e-8)
            grad_norm = grad_magnitude / (tf.reduce_max(grad_magnitude) + 1e-8)

            # Apply denoising step
            x = x - weight * grad_norm

        return tf.clip_by_value(x, 0.0, 255.0)

    _fn.__name__ = "total_variation_denoise_tf"
    return _fn


# Non-Local Means Denoiser (BPDA-friendly)
def nlm_denoise_bpda(h: float = 0.1, patch_size: int = 3, window_size: int = 7, step: int = 2):
    """
    Create BPDA-friendly Non-Local Means denoiser.
    """
    @tf.custom_gradient
    def _fn(x: tf.Tensor) -> tf.Tensor:
        x_dtype = x.dtype
        x_f = tf.cast(x, tf.float32) / 255.0

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
        pad_r = (window_size - 1) // 2

        for dy in tf.range(-pad_r, pad_r + 1, step):
            for dx in tf.range(-pad_r, pad_r + 1, step):
                patches_shifted = tf.roll(patches, shift=[dy, dx], axis=[1, 2])
                img_shifted = tf.roll(x_f, shift=[dy, dx], axis=[1, 2])
                dist2 = tf.reduce_sum(tf.square(center - patches_shifted), axis=-1, keepdims=True)
                w = tf.exp(-dist2 / (h_sq + EPS))
                weighted_sum += w * img_shifted
                weight_sum += w

        denoised = weighted_sum / (weight_sum + EPS)
        denoised_scaled = tf.clip_by_value(denoised * 255.0, 0.0, 255.0)
        denoised_scaled = tf.cast(denoised_scaled, x_dtype)

        def grad(upstream):
            grad_scale = tf.constant(0.7, dtype=upstream.dtype)
            return grad_scale * upstream

        return denoised_scaled, grad
    _fn.__name__ = "nlm_denoise_bpda"
    return _fn
