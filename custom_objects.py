from loss import disparity_loss_function, reconstruction_loss_function
from layers import BilinearUpSampling2D
import tensorflow as tf
from bilinear_sampler import generate_image_left, generate_image_right

custom_objects = {
            'BilinearUpSampling2D': BilinearUpSampling2D, 
            'disparity_loss_function': disparity_loss_function,
            'reconstruction_loss_function': reconstruction_loss_function,
            'tf': tf,
            'generate_image_left': generate_image_left,
            'generate_image_right': generate_image_right,
            }