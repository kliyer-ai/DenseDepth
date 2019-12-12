from loss import disparity_loss_function, reconstruction_loss_function
from layers import BilinearUpSampling2D

custom_objects = {
            'BilinearUpSampling2D': BilinearUpSampling2D, 
            'disparity_loss_function': disparity_loss_function,
            'reconstruction_loss_function': reconstruction_loss_function
            }