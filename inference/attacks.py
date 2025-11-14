import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent
import numpy as np

def generate_adversarial_examples_trades(
    model, x, y, epsilon, step_size, steps, preprocess_fn=None
):
    """
    Generate adversarial examples adaptively (BPDA-aware).
    Based on TRADES attack objective (KL divergence).
    """
    x_adv = x + 0.001 * tf.random.normal(tf.shape(x), dtype=tf.float32)
    x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)
    inp_nat = preprocess_fn(x) if preprocess_fn != None else x
    logits_nat = model(inp_nat, training=False)
    logits_nat = tf.stop_gradient(logits_nat)

    x_adv_var = tf.Variable(x_adv)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv_var)
            inp_adv = preprocess_fn(x_adv_var) if preprocess_fn != None else x_adv_var
            logits_adv = model(inp_adv, training=False)
            p = tf.nn.softmax(logits_nat, axis=-1)
            q = tf.nn.softmax(logits_adv, axis=-1)
            p = tf.clip_by_value(p, 1e-12, 1.0)
            q = tf.clip_by_value(q, 1e-12, 1.0)
            kl = tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=-1)
            loss_kl = tf.reduce_mean(kl)
        grad = tape.gradient(loss_kl, x_adv_var)
        x_adv_var.assign_add(step_size * tf.sign(grad))
        x_adv_var.assign(tf.clip_by_value(x_adv_var, x - epsilon, x + epsilon))
        x_adv_var.assign(tf.clip_by_value(x_adv_var, 0.0, 255.0))

    return tf.stop_gradient(x_adv_var)


def generate_adversarial_examples_PGD(model, x, epsilon, step_size, pertub_steps):
    if isinstance(x, tf.Tensor):
        x_np = x.numpy()
    else:
        x_np = np.array(x)

    x_np = x_np.astype(np.float32)

    classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=4,
        input_shape=(224, 224, 3),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        clip_values=(0.0, 255.0) 
    )

    atk = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=epsilon,
        eps_step=step_size,
        max_iter=pertub_steps,
    )

    x_adv = atk.generate(x_np)
    return x_adv
