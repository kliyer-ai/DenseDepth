import io
import random
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from utils import predict, evaluate

import tensorflow as tf

def make_image(tensor):
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

def get_callbacks(model, basemodel, train_generator, test_generator, runPath, test_set):
    callbacks = []

    # Callback: Tensorboard
    class LRTensorBoard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir, **kwargs):
            super().__init__(log_dir=log_dir, **kwargs)

            self.num_samples = 6
            self.train_idx = np.random.randint(low=0, high=len(train_generator), size=10)
            self.test_idx = np.random.randint(low=0, high=len(test_generator), size=10)

        def on_epoch_end(self, epoch, logs=None):            
            if not test_set == None:
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

                    xs_train = list(map(lambda x: x[0], xs_train))
                    xs_test = list(map(lambda x: x[0], xs_test))

                    y_train = np.clip(y_train[0], minDepth, maxDepth) / maxDepth 
                    y_test = np.clip(y_test[0], minDepth, maxDepth) / maxDepth 

                    h, w = y_train.shape[0], y_train.shape[1]

                    rgb_train = list(map(lambda x: resize(x, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True), xs_train))
                    rgb_test = list(map(lambda x: resize(x, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True), xs_test))

                    gt_train = plasma(y_train[:,:,0])[:,:,:3]
                    gt_test = plasma(y_test[:,:,0])[:,:,:3]

                    predict_train = plasma(predict(model, xs_train)[0,:,:,0])[:,:,:3]
                    predict_test = plasma(predict(model, xs_test)[0,:,:,0])[:,:,:3]

                    train_samples.append(np.vstack(rgb_train + [gt_train, predict_train]))
                    test_samples.append(np.vstack(rgb_test + [gt_test, predict_test]))

                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_image(255 * np.hstack(train_samples)))]), epoch)
                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_image(255 * np.hstack(test_samples)))]), epoch)

                # Metrics
                # e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True)
                # logs.update({'rel': e[3]})
                # logs.update({'rms': e[4]})
                # logs.update({'log10': e[5]})

            super().on_epoch_end(epoch, logs)
    callbacks.append( LRTensorBoard(log_dir=runPath, histogram_freq=0, write_graph=False, write_grads=False) )

    # Callback: Learning Rate Scheduler
    # lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00001, min_delta=1e-2, verbose=1)
    # callbacks.append( lr_schedule ) # reduce learning rate when stuck

    # Callback: save checkpoints
    # callbacks.append(keras.callbacks.ModelCheckpoint(runPath + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
    #     verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=5))

    return callbacks