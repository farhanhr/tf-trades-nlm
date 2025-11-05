import tensorflow as tf
from typing import Literal, Callable

EPS = 1e-12

def _kl_divergence_per_sample(logits_p, logits_q):
    """KL(p || q) per-sample (sum over classes) -> shape (batch,)."""
    p = tf.nn.softmax(logits_p, axis=-1)
    q = tf.nn.softmax(logits_q, axis=-1)
    p = tf.clip_by_value(p, 1e-12, 1.0)
    q = tf.clip_by_value(q, 1e-12, 1.0)
    kl = tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=-1)
    return kl

def trades_loss_for_vgg(
    model: tf.keras.Model,
    x_natural_raw: tf.Tensor,
    y: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
    step_size: float = 2.0,
    epsilon: float = 8.0,
    perturb_steps: int = 10,
    beta: float = 1.0,
    distance: Literal["l_inf", "l_2"] = "l_inf",
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] = None,
):
    """
    TRADES loss for models expecting VGG preprocess. Works on raw pixel inputs [0,255].
    - x_natural_raw: float32 tensor, values in [0,255]
    - step_size, epsilon: pixel units (e.g., 2.0, 8.0)
    - preprocess_fn: function that maps raw pixels -> model input (e.g. vgg16.preprocess_input)
        Must be applied INSIDE gradient tapes when computing logits from x_adv.
    Returns scalar loss tensor; also applies optimizer update to model weights.
    """

    x_natural_raw = tf.cast(x_natural_raw, tf.float32)
    y = tf.cast(tf.reshape(y, (-1,)), tf.int32)
    batch_size = tf.shape(x_natural_raw)[0]

    # --- 1) create x_adv in raw pixel space ---
    x_adv = x_natural_raw + 0.001 * tf.random.normal(tf.shape(x_natural_raw), dtype=tf.float32)
    x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)

    # compute logits_nat once (preprocess then model), detach
    inp_nat_for_model = preprocess_fn(x_natural_raw) if preprocess_fn is not None else x_natural_raw
    logits_nat = model(inp_nat_for_model, training=False)
    logits_nat = tf.stop_gradient(logits_nat)

    if distance == "l_inf":
        for _ in range(int(perturb_steps)):
            x_adv_var = tf.Variable(x_adv)
            with tf.GradientTape() as tape:
                tape.watch(x_adv_var)
                inp_adv = preprocess_fn(x_adv_var) if preprocess_fn is not None else x_adv_var
                logits_adv = model(inp_adv, training=False)
                kl_per = _kl_divergence_per_sample(logits_nat, logits_adv)
                loss_kl = tf.reduce_mean(kl_per)
            grad = tape.gradient(loss_kl, x_adv_var)
            x_adv = x_adv + step_size * tf.sign(grad)
            x_adv = tf.clip_by_value(x_adv, x_natural_raw - epsilon, x_natural_raw + epsilon)
            x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)
        x_adv = tf.stop_gradient(x_adv)

    elif distance == "l_2":
        delta = x_adv - x_natural_raw
        delta = tf.Variable(delta)
        for _ in range(int(perturb_steps)):
            with tf.GradientTape() as tape:
                tape.watch(delta)
                adv = x_natural_raw + delta
                adv = tf.clip_by_value(adv, 0.0, 255.0)
                inp_adv = preprocess_fn(adv) if preprocess_fn is not None else adv
                logits_adv = model(inp_adv, training=False)
                kl_per = _kl_divergence_per_sample(logits_nat, logits_adv)
                loss_kl = tf.reduce_mean(kl_per)
            g = tape.gradient(loss_kl, delta)
            # normalize per-sample
            g_flat = tf.reshape(g, [tf.shape(g)[0], -1])
            g_norm = tf.norm(g_flat, axis=1, keepdims=True)
            g_norm_safe = tf.maximum(g_norm, EPS)
            g_unit = g / tf.reshape(g_norm_safe, tf.concat([[tf.shape(g)[0]], tf.ones(tf.rank(g)-1, tf.int32)], axis=0))
            delta.assign_add(step_size * g_unit)
            # project to L2 ball
            delta_flat = tf.reshape(delta, [tf.shape(delta)[0], -1])
            delta_norm = tf.norm(delta_flat, axis=1, keepdims=True)
            factor = tf.minimum(1.0, epsilon / tf.maximum(delta_norm, EPS))
            delta.assign(tf.reshape(delta_flat * factor, tf.shape(delta)))
            delta.assign(tf.clip_by_value(x_natural_raw + delta, 0.0, 255.0) - x_natural_raw)
        x_adv = tf.stop_gradient(x_natural_raw + delta)
    else:
        x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)
        x_adv = tf.stop_gradient(x_adv)

    # --- 2) compute TRADES loss and update model ---
    with tf.GradientTape() as tape:
        inp_nat = preprocess_fn(x_natural_raw) if preprocess_fn is not None else x_natural_raw
        inp_adv = preprocess_fn(x_adv) if preprocess_fn is not None else x_adv

        logits = model(inp_nat, training=True)
        logits_adv = model(inp_adv, training=True)

        loss_natural = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        kl_per = _kl_divergence_per_sample(logits, logits_adv)
        loss_robust = tf.reduce_mean(kl_per)
        loss = loss_natural + beta * loss_robust

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss
