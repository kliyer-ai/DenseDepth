import keras.backend as K
import tensorflow as tf

def edges(y_true, y_pred):
    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    return K.mean(l_edges)

def point_wise_depth(y_true, y_pred):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
    return 0.1 * K.mean(l_depth)

def ssim(y_true, y_pred):
    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)
    return l_ssim