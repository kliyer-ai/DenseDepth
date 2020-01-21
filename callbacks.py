import io
import random
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from utils import predict, evaluate, compute_errors, scale_up

import tensorflow as tf

def make_image(tensor):
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

def get_callbacks(model, basemodel, train_generator, test_generator, runPath):
    callbacks = []

    # Callback: Tensorboard
    class LRTensorBoard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir, **kwargs):
            super().__init__(log_dir=log_dir, **kwargs)

            self.num_samples = 6
            self.train_idx =  np.random.randint(low=0, high=len(train_generator), size=self.num_samples)
            # always get same test samples
            self.test_idx = np.array(range(self.num_samples)) * (len(test_generator) // self.num_samples) # np.random.randint(low=0, high=len(test_generator), size=10)

        def on_epoch_end(self, epoch, logs=None):            
            # Samples using current model
            import matplotlib.pyplot as plt
            from skimage.transform import resize
            plasma = plt.get_cmap('Greys_r')

            minDepth, maxDepth = 0, 1

            train_samples = []
            test_samples = []

            for i in range(self.num_samples):
                xs_train, y_train = train_generator.__getitem__(self.train_idx[i], False)
                xs_test, y_test = test_generator[self.test_idx[i]]

        	    # get first batch
                xs_train = list(map(lambda x: x[0], xs_train))
                xs_test = list(map(lambda x: x[0], xs_test))
                y_train = list(map(lambda x: x[0], y_train))
                y_test = list(map(lambda x: x[0], y_test))

                # onlt get disps, second arg is images again
                disps_train, _ = y_train
                disps_test, _  = y_test

                h, w = disps_train[0].shape[0], disps_train[0].shape[1]

                rgb_train = list(map(lambda x: resize(x, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True), xs_train[:2]))
                rgb_test = list(map(lambda x: resize(x, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True), xs_test[:2]))

                gt_train = list(map(lambda x: plasma(x[:,:,0])[:,:,:3], disps_train)) 
                gt_test = list(map(lambda x: plasma(x[:,:,0])[:,:,:3], disps_test))
                
                disps_pred_train, reconstructions_train = predict(model, xs_train)
                reconstructions_train = list(reconstructions_train[0])
                predict_trains = list(map(lambda x: plasma(x[:,:,0])[:,:,:3], disps_pred_train[0])) 

                disps_pred_test, reconstructions_test = predict(model, xs_test)
                reconstructions_test = list(reconstructions_test[0])
                predicts_test = list(map(lambda x: plasma(x[:,:,0])[:,:,:3], disps_pred_test[0])) 
                
                train_samples.append(np.vstack(rgb_train + reconstructions_train + gt_train + predict_trains))
                test_samples.append(np.vstack(rgb_test + reconstructions_test + gt_test + predicts_test))

            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_image(255 * np.hstack(train_samples)))]), epoch)
            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_image(255 * np.hstack(test_samples)))]), epoch)

            # Metrics
            e = evaluate(model, test_generator)
            
            # e = compute_errors(disps_test[0], disps_pred_test[0,0]) # FIX!!
            logs.update({'rel': e[3]})
            logs.update({'rms': e[4]})
            logs.update({'log10': e[5]})

            super().on_epoch_end(epoch, logs)
    callbacks.append( LRTensorBoard(log_dir=runPath, histogram_freq=0, write_graph=False, write_grads=True) )

    # Callback: Learning Rate Scheduler
    # lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00001, min_delta=1e-2, verbose=1)
    # callbacks.append( lr_schedule ) # reduce learning rate when stuck

    # Callback: save checkpoints
    # callbacks.append(keras.callbacks.ModelCheckpoint(runPath + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
    #     verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=5))

    return callbacks