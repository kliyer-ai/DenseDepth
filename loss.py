import keras.backend as K
import tensorflow as tf
from bilinear_sampler import generate_image_left, generate_image_right
from shape import get_shape_rgb

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

def get_disparity_smoothness(disp):
    disp_gradients_y, disp_gradients_x = tf.image.image_gradients(disp)
    total_var = disp_gradients_x + disp_gradients_y
    return total_var

# =============================================================================================

def supervised_loss_function(y_true, y_pred):
    return ssim(y_true, y_pred) + edges(y_true, y_pred) + 0.1 * point_wise_depth(y_true, y_pred)

def disparity_loss_function(y_true, y_pred):
    left_disp = y_true[:, 0]
    right_disp = y_true[:, 1]
    left_mask = tf.math.equal(left_disp, 0.0)
    right_mask = tf.math.equal(right_disp, 0.0)

    left_disp_est, right_disp_est = tf.unstack(y_pred, axis=1)
    # only keeps the estimated disparity for the masked regions, i.e. the wires
    masked_left_disp_est = tf.where(left_mask, tf.zeros(tf.shape(left_disp)), left_disp_est)
    masked_right_disp_est = tf.where(right_mask, tf.zeros(tf.shape(right_disp)), right_disp_est)
    
    # LR CONSISTENCY
    right_to_left_disp = generate_image_left(right_disp_est, left_disp_est)
    left_to_right_disp = generate_image_right(left_disp_est, right_disp_est)

    lr_left_loss = tf.reduce_mean(tf.abs(right_to_left_disp - left_disp_est))
    lr_right_loss = tf.reduce_mean(tf.abs(left_to_right_disp - right_disp_est))
    total_lr_loss = lr_left_loss + lr_right_loss

    # DISPARITY SMOOTHNESS
    # Acts as regularization term
    disp_left_loss  = tf.reduce_mean(tf.abs( get_disparity_smoothness(left_disp_est) )) 
    disp_right_loss = tf.reduce_mean(tf.abs( get_disparity_smoothness(right_disp_est) ))
    disp_gradient_loss = disp_left_loss + disp_right_loss

    # SUPERVISED loss
    sup_left_loss = supervised_loss_function(left_disp, masked_left_disp_est) 
    sup_right_loss = supervised_loss_function(right_disp, masked_right_disp_est) 
    total_sup_loss = sup_left_loss + sup_right_loss

    return 1.0 * total_lr_loss + 0.1 * disp_gradient_loss + 1.0 * total_sup_loss

# image reconstruction
# l1 and ssim
def reconstruction_loss_function(y_true, y_pred):
    left_image = y_true[:, 0]
    right_image = y_true[:, 1]
    left_recon, right_recon = tf.unstack(y_pred, axis=1)

    # CROP
    height, width, channels = get_shape_rgb()
    crop_width = 0.7 * width
    left_image_c = tf.image.resize_image_with_crop_or_pad(left_image, height, crop_width)
    right_image_c = tf.image.resize_image_with_crop_or_pad(right_image, height, crop_width)
    left_recon_c = tf.image.resize_image_with_crop_or_pad(left_recon, height, crop_width)
    right_recon_c = tf.image.resize_image_with_crop_or_pad(right_recon, height, crop_width)

    # L1
    left_l1 = point_wise_depth(left_image_c, left_recon_c)
    right_l1 = point_wise_depth(right_image_c, right_recon_c)
    total_l1 = left_l1 + right_l1

    # SSIM
    left_ssim = ssim(left_image_c, left_recon_c)
    right_ssim = ssim(right_image_c, right_recon_c)
    total_ssim = left_ssim + right_ssim

    l1_weight = 0.85
    image_loss = l1_weight * total_l1 + (1 - l1_weight) * total_ssim
    return image_loss
