import tensorflow as tf
from bilinear_sampler import generate_image_left, generate_image_right
from shape import get_shape_rgb

def edges(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true)) 

def l1_loss(y_true, y_pred=None):
    return tf.reduce_mean(tf.abs(y_pred - y_true)) 

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

def get_weighted_disparity_smoothness(disp, img):
    disp_height = tf.shape(disp)[1]
    disp_width = tf.shape(disp)[2]

    img = tf.image.resize_area(img, [disp_height, disp_width], align_corners=True)
    img_dy, img_dx = tf.image.image_gradients(img)

    # large gradient means small weight
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), 3, keepdims=True)) 
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), 3, keepdims=True)) 

    disp_dy, disp_dx = tf.image.image_gradients(disp)
    # disp_loss = tf.reduce_mean(tf.abs( weights_y * disp_dy + weights_x disp_dx )) 

    return weights_y * disp_dy + weights_x * disp_dx

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

# alternative crop functions
# crops based on individual max disparity values. As this results in images of different dimensions, images get resized to the same dims
# does not work very well; probably due to interpolation 
def crop_l(img, num_disp):
    height       = tf.shape(img)[1]
    width        = tf.shape(img)[2]
    width_f      = tf.cast(width, tf.float32)
    boxes = tf.map_fn(lambda l_disp: [0.0, l_disp[0]/width_f, 1.0, 1.0], num_disp, dtype=[tf.float32, tf.float32, tf.float32, tf.float32])
    return tf.image.crop_and_resize(img, boxes, [0,1,2,3], [height, width])


def crop_r(img, num_disp):
    height       = tf.shape(img)[1]
    width        = tf.shape(img)[2]
    width_f      = tf.cast(width, tf.float32)
    boxes = tf.map_fn(lambda l_disp: [0, 0, 1, (width_f - l_disp[0]) / width_f], num_disp, dtype=[tf.float32, tf.float32, tf.float32, tf.float32])
    return tf.image.crop_and_resize(img, boxes, [0,1,2,3], [height, width])

# =============================================================================================

def supervised_loss_function(y_true, y_pred, include_edges=True):
    # edges loss ensures that wires are smooth
    alpha_edges = 1 if include_edges else 0
    return image_loss(y_true, y_pred) + alpha_edges * edges(y_true, y_pred)

def disparity_loss_function(y_true, y_pred, crop_factor=0.6):
    supervised_weight = 1.0
    disp_weight = 1.0

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
        reg_left_loss = tf.reduce_mean(tf.abs(crop_left(left_disp_est, crop_factor))) 
        reg_right_loss = tf.reduce_mean(tf.abs(crop_right(right_disp_est, crop_factor))) 
    else:
        lr_left_loss = tf.reduce_mean(tf.abs(right_to_left_disp - left_disp_est))
        lr_right_loss = tf.reduce_mean(tf.abs(left_to_right_disp - right_disp_est))
        reg_left_loss = 0.0 
        reg_right_loss = 0.0  
    total_lr_loss = lr_left_loss + lr_right_loss

    # DISPARITY SMOOTHNESS
    left_image = graph.get_tensor_by_name('input_1:0')
    disp_left_loss  = tf.reduce_mean(tf.abs( get_weighted_disparity_smoothness(left_disp_est, left_image) )) 

    right_image = graph.get_tensor_by_name('input_2:0')
    disp_right_loss = tf.reduce_mean(tf.abs( get_weighted_disparity_smoothness(right_disp_est, right_image) ))

    disp_gradient_loss = disp_left_loss + disp_right_loss


    # REGULARIZATION  
    reg_total_loss = reg_left_loss + reg_right_loss

    # TOTAL SELF SUPERVISED LOSS
    total_disp_loss = 1.0 * total_lr_loss + 0.1 * disp_gradient_loss + 0.05 * reg_total_loss


    # SUPERVISED LOSS
    mask = supervised_weight > 0 and disp_weight > 0 

    left_mask = tf.math.equal(left_disp, 0.0)
    right_mask = tf.math.equal(right_disp, 0.0)
    # only keeps the estimated disparity for the masked regions, i.e. the wires
    masked_left_disp_est = tf.where(left_mask, tf.zeros(tf.shape(left_disp)), left_disp_est) if mask else left_disp_est
    masked_right_disp_est = tf.where(right_mask, tf.zeros(tf.shape(right_disp)), right_disp_est) if mask else right_disp_est
    sup_left_loss = supervised_loss_function(left_disp, masked_left_disp_est, include_edges=not mask) 
    sup_right_loss = supervised_loss_function(right_disp, masked_right_disp_est, include_edges=not mask) 
    total_sup_loss = sup_left_loss + sup_right_loss

    
    return disp_weight * total_disp_loss + supervised_weight * total_sup_loss

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
    return 1.0 * total_loss