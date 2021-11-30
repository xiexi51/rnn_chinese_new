import tensorflow as tf
import numpy as np

def exp_safe(x):
    return tf.clip_by_value(tf.exp(x), clip_value_min=1e-10, clip_value_max=1e10)

def log_safe(x):
    return tf.clip_by_value(tf.math.log(x), clip_value_min=-20, clip_value_max=20)

def N(x, mu, sigma):
    return exp_safe(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def Loss(y, pred):
    tf.debugging.assert_all_finite(pred, 'loss inputs ill')
    pi, mux, muy, sigmax, sigmay,  = tf.split(pred[:, :, :-3], 5, axis=-1)
    p = pred[:, :, -3:]
    xtp1 = tf.expand_dims(y[:, :, 0], axis=-1)
    ytp1 = tf.expand_dims(y[:, :, 1], axis=-1)
    stp1 = y[:, :, 2:5]
    w = tf.constant([1, 5, 100], dtype=tf.float32)
    lPd = log_safe(tf.reduce_sum(pi * N(xtp1, mux, sigmax) * N(ytp1, muy, sigmay), axis=-1))
    lPs = tf.reduce_sum(w * stp1 * log_safe(p), axis=-1)
    return - (lPd + lPs)