import tensorflow as tf
from bilinear_sampler import generate_image_left, generate_image_right
from shape import get_shape_rgb

def edges(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true)) 

def l1_loss(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_pred - y_true)) 
    return l1

def ssim(y_true, y_pred):
    return tf.clip_by_value((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0.0, 1.0)

def image_loss(y_true, y_pred):
    l1 = l1_loss(y_true, y_pred)
    s = ssim(y_true, y_pred)
    ssim_weight = 0.9
    image_loss = ssim_weight * s + (1 - ssim_weight) * l1
    return image_loss

def get_disparity_smoothness(disp):
    disp_gradients_y, disp_gradients_x = tf.image.image_gradients(disp)
    total_var = disp_gradients_x + disp_gradients_y
    return total_var

def crop_left(img, crop_factor):
    height       = tf.shape(img)[1]
    width        = tf.shape(img)[2]
    width_f      = tf.cast(width, tf.float32)
    crop_width   = tf.cast(crop_factor * width_f, tf.int32)
    return tf.image.crop_to_bounding_box(img, 0, width - crop_width, height, crop_width)

def crop_right(img, crop_factor):
    height       = tf.shape(img)[1]
    width        = tf.shape(img)[2]
    width_f      = tf.cast(width, tf.float32)
    crop_width   = tf.cast(crop_factor * width_f, tf.int32)
    return tf.image.crop_to_bounding_box(img, 0, 0, height, crop_width)

# =============================================================================================

def supervised_loss_function(y_true, y_pred):
    # edges loss ensures that wires are smooth
    return image_loss(y_true, y_pred) + 1 * edges(y_true, y_pred)

def disparity_loss_function(y_true, y_pred, crop_factor=0.6, mask=False):
    left_disp = y_true[:, 0]
    right_disp = y_true[:, 1]

    graph = tf.get_default_graph()
    num_disp = graph.get_tensor_by_name('input_3:0')

    left_disp_est, right_disp_est = tf.unstack(y_pred, axis=1)
    
    # LR CONSISTENCY
    right_to_left_disp = generate_image_left(right_disp_est, left_disp_est, num_disp, resize=False)
    left_to_right_disp = generate_image_right(left_disp_est, right_disp_est, num_disp, resize=False)

    # OPTIONAL CROP
    if crop_factor < 1.0:
        lr_left_loss = tf.reduce_mean(tf.abs(crop_left(right_to_left_disp, crop_factor) - crop_left(left_disp_est, crop_factor)))
        lr_right_loss = tf.reduce_mean(tf.abs(crop_right(left_to_right_disp, crop_factor) - crop_right(right_disp_est, crop_factor)))
    else:
        lr_left_loss = tf.reduce_mean(tf.abs(right_to_left_disp - left_disp_est))
        lr_right_loss = tf.reduce_mean(tf.abs(left_to_right_disp - right_disp_est))
    total_lr_loss = lr_left_loss + lr_right_loss

    # DISPARITY SMOOTHNESS
    disp_left_loss  = tf.reduce_mean(tf.abs( get_disparity_smoothness(left_disp_est) )) 
    disp_right_loss = tf.reduce_mean(tf.abs( get_disparity_smoothness(right_disp_est) ))
    disp_gradient_loss = disp_left_loss + disp_right_loss

    # TOTAL SELF SUPERVISED LOSS
    total_disp_loss = 1.0 * total_lr_loss + 0.1 * disp_gradient_loss


    # SUPERVISED LOSS
    left_mask = tf.math.equal(left_disp, 0.0)
    right_mask = tf.math.equal(right_disp, 0.0)
    # only keeps the estimated disparity for the masked regions, i.e. the wires
    masked_left_disp_est = tf.where(left_mask, tf.zeros(tf.shape(left_disp)), left_disp_est) if mask else left_disp_est
    masked_right_disp_est = tf.where(right_mask, tf.zeros(tf.shape(right_disp)), right_disp_est) if mask else right_disp_est
    sup_left_loss = supervised_loss_function(left_disp, masked_left_disp_est) 
    sup_right_loss = supervised_loss_function(right_disp, masked_right_disp_est) 
    total_sup_loss = sup_left_loss + sup_right_loss

    supervised_weight = 1.0 #2.0
    return 0 * total_disp_loss + supervised_weight * total_sup_loss

# image reconstruction
# l1 and ssim
def reconstruction_loss_function(y_true, y_pred, crop_factor=0.6):
    left_image = y_true[:, 0]
    right_image = y_true[:, 1]
    left_recon, right_recon = tf.unstack(y_pred, axis=1)

    # OPTIONAL CROP
    if crop_factor < 1.0:
        left_image = crop_left(left_image, crop_factor)
        right_image = crop_right(right_image, crop_factor)
        left_recon = crop_left(left_recon, crop_factor)
        right_recon = crop_right(right_recon, crop_factor)

    left_image_loss = image_loss(left_image, left_recon)
    right_image_loss = image_loss(right_image, right_recon)
    total_loss = left_image_loss + right_image_loss
    return 0 * total_loss