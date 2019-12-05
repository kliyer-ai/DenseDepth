import keras.backend as K
import tensorflow as tf

def edges(y_true, y_pred):
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
        return K.mean(l_edges)

def point_wise_depth(y_true, y_pred):
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
    return K.mean(l_depth)

def ssim(y_true, y_pred):
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)
    return 
    
# =============================================================================================

def depth_loss_function(y_true, y_pred):
    return ssim(y_true, y_pred) + edges(y_true, y_pred) + 0.1 * point_wise_depth(y_true, y_pred)

def reconstruction_loss_function(y_true, y_pred):
    l1_weight = 0.85
    return l1_weight * point_wise_depth(y_true, y_pred) + (1 - l1_weight) * ssim(y_true, y_pred)