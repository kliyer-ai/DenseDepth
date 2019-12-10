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
    return K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)
    
# =============================================================================================

def depth_loss_function(y_true, y_pred):
    print('================================')
    print(y_true)
    print(y_pred)
    left_y_true = y_true
    left_y_pred, right_y_pred = tf.unstack(y_pred, axis=1)
    print(left_y_pred.shape)
    print('true',left_y_true.shape)
    return ssim(left_y_true, left_y_pred) + edges(left_y_true, left_y_pred) + 0.1 * point_wise_depth(left_y_true, left_y_pred)

def reconstruction_loss_function(y_true, y_pred):
    left_y_true = y_true[:,0]
    right_y_true = y_true[:,1]
    print('recon',left_y_true.shape)

    left_y_pred, right_y_pred = tf.unstack(y_pred, axis=1)
    # right_y_pred = y_pred[1]

    l1_weight = 0.85
    return l1_weight * point_wise_depth(left_y_true, left_y_pred) + (1 - l1_weight) * ssim(left_y_true, left_y_pred)