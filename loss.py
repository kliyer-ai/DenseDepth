import keras.backend as K
import tensorflow as tf
from bilinear_sampler import generate_image_left, generate_image_right


def edges(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) +
                     K.abs(dx_pred - dx_true), axis=-1)
    return K.mean(l_edges)


def point_wise_depth(y_true, y_pred):
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
    return K.mean(l_depth)


def ssim(y_true, y_pred):
    return K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)


# LSIM
def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def get_total_var(disp):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)
    total_var = disp_gradients_x + disp_gradients_y
    return total_var

# =============================================================================================


def depth_loss_function(y_true, y_pred):
    # left_disp = y_true[:, 0]
    # right_disp = y_true[:, 1]
    # left_disp, right_disp = tf.unstack(y_true, axis=1)
    left_disp_est, right_disp_est = tf.unstack(y_pred, axis=1)
    
    # LR CONSISTENCY
    right_to_left_disp = generate_image_left(right_disp_est, left_disp_est)
    left_to_right_disp = generate_image_right(left_disp_est, right_disp_est)

    lr_left_loss = tf.reduce_mean(tf.abs(right_to_left_disp - left_disp_est))
    lr_right_loss = tf.reduce_mean(tf.abs(left_to_right_disp - right_disp_est))
    total_lr_loss = lr_left_loss + lr_right_loss

    # DISPARITY SMOOTHNESS
    disp_left_loss  = tf.reduce_mean(tf.abs( get_total_var(left_disp_est) )) 
    disp_right_loss = tf.reduce_mean(tf.abs( get_total_var(right_disp_est) ))
    disp_gradient_loss = disp_left_loss + disp_right_loss

    # ssim(left_disp, left_y_pred) + edges(left_disp, left_y_pred) + 0.1 * point_wise_depth(left_disp, left_y_pred)

    return 1.0 * total_lr_loss + 0.1 * disp_gradient_loss

# image reconstruction
# l1 and ssim
def reconstruction_loss_function(y_true, y_pred):
    left_image = y_true[:, 0]
    right_image = y_true[:, 1]
    left_recon, right_recon = tf.unstack(y_pred, axis=1)

    # L1
    left_l1 = point_wise_depth(left_image, left_recon)
    right_l1 = point_wise_depth(right_image, right_recon)
    total_l1 = left_l1 + right_l1

    # SSIM
    left_ssim = ssim(left_image, left_recon)
    right_ssim = ssim(right_image, right_recon)
    total_ssim = left_ssim + right_ssim

    l1_weight = 0.85
    return l1_weight * total_l1 + (1 - l1_weight) * total_ssim
