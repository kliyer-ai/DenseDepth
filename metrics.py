import keras.backend as K
import tensorflow as tf

def rmse(y_true, y_pred):
    rmse = K.square(y_true - y_pred) 
    return K.sqrt(K.mean(rmse))

def log_10(y_true, y_pred):
    log_10 = K.abs(K.log(y_true) - K.log(y_pred))
    return K.mean(log_10)

def delta_1(y_true, y_pred):
    thresh = K.max((y_true / y_pred), (y_pred / y_true))
    return K.mean(thresh < 1.25)

def delta_2(y_true, y_pred):
    thresh = K.max((y_true / y_pred), (y_pred / y_true))
    return K.mean(thresh < 1.25 ** 2)

def delta_3(y_true, y_pred):
    thresh = K.max((y_true / y_pred), (y_pred / y_true))
    return K.mean(thresh < 1.25 ** 3)

def abs_rel(y_true, y_pred):
    return K.mean(K.abs((y_true - y_pred) / y_true))

