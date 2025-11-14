import tensorflow as tf
from typing import Callable

EPS = 1e-12

@tf.function
def kl_div(logits_p, logits_q):
    """KL(p || q) with batchmean reduction."""
    p = tf.nn.softmax(logits_p, axis=-1)
    log_p = tf.nn.log_softmax(logits_p, axis=-1)
    log_q = tf.nn.log_softmax(logits_q, axis=-1)
    kl = tf.reduce_sum(p * (log_p - log_q), axis=1)
    return tf.reduce_mean(kl)

def trades_loss_tf(
    model: tf.keras.Model,
    x_natural: tf.Tensor,        # input before preprocess_input
    y: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
    step_size=2.0, #(x/255)
    epsilon=8.0, #(x/255)
    perturb_steps=10,
    beta=6.0,
    preprocess_fn: Callable = None,
):
    x_natural = tf.cast(x_natural, tf.float32)
    y = tf.cast(tf.reshape(y, (-1,)), tf.int32)

    # Compute natural logits (detach)
    x_nat_pre = preprocess_fn(x_natural) if preprocess_fn else x_natural
    logits_nat = model(x_nat_pre, training=False)
    logits_nat_detach = tf.stop_gradient(logits_nat)

    # Adversarial Example Generation (PGD)
    x_adv = x_natural + 0.001 * tf.random.normal(tf.shape(x_natural))
    # x_adv = x_natural + tf.random.uniform(tf.shape(x_natural), -epsilon, epsilon)
    
    for _ in range(perturb_steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            x_adv_pre = preprocess_fn(x_adv) if preprocess_fn else x_adv
            logits_adv = model(x_adv_pre, training=False)
            loss_kl = kl_div(logits_nat_detach, logits_adv)

        grad = tape.gradient(loss_kl, x_adv)
        x_adv = x_adv + step_size * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x_natural - epsilon, x_natural + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)

    # Compute TRADES Loss
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)

        # forward pass
        x_nat_pre = preprocess_fn(x_natural) if preprocess_fn else x_natural
        x_adv_pre = preprocess_fn(x_adv) if preprocess_fn else x_adv

        logits_nat = model(x_nat_pre, training=True)
        logits_adv = model(x_adv_pre, training=True)

        # CE loss
        loss_nat = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_nat)
        )
        # KL loss
        loss_robust = kl_div(logits_nat, logits_adv)
        loss_total = loss_nat + beta * loss_robust

    grads = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_total, loss_nat, loss_robust

