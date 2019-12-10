import numpy as np
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy

from utils import resize

from shape import get_shape_rgb, get_shape_depth

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def get_data(batch_size, data_zipfile):
    data = extract_zip(data_zipfile)

    train = list((row.split(',') for row in (data['data/train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    test = list((row.split(',') for row in (data['data/test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    # Helpful for testing...
    if False:
        train = train[:10]
        test = test[:10]

    return data, train, test

def get_train_test_data(batch_size, data_zipfile, nr_inputs=1):
    data, train, test = get_data(batch_size, data_zipfile)

    train_generator = BasicRGBSequence(data, train, batch_size, nr_inputs=nr_inputs, train=True)
    test_generator = BasicRGBSequence(data, test, batch_size, nr_inputs=nr_inputs)

    return train_generator, test_generator

class BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, nr_inputs=1, train=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.0, flip_ratio=0.50, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=0.0 if not is_erase else 0.50)
        self.batch_size = batch_size
        self.train = train
        self.nr_inputs = nr_inputs

        from sklearn.utils import shuffle
        if self.train:
            self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=False):
        disp_is_halved = False

        disp_ph = np.zeros(get_shape_depth(batch_size=self.batch_size, halved=disp_is_halved))
        left_image_ph = np.zeros(get_shape_rgb(batch_size=self.batch_size))
        right_image_ph = np.zeros(get_shape_rgb(batch_size=self.batch_size))

        batch_x = [left_image_ph, right_image_ph]
        batch_y = [np.stack([disp_ph, disp_ph.copy()], axis=1), np.stack([left_image_ph.copy(), right_image_ph.copy()], axis=1)]

        # Augmentation of RGB images
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)
            sample = self.dataset[index]

            left_image = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(get_shape_rgb())/255,0,1)
            left_image = resize(left_image, get_shape_rgb()[1])
            right_image = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(get_shape_rgb())/255,0,1)
            right_image = resize(right_image, get_shape_rgb()[1])

            disparity = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[-2]]) )).reshape(get_shape_depth(halved=disp_is_halved))/255,0,1)
            disparity = resize(disparity, get_shape_depth(halved=disp_is_halved)[1])

            if self.train and is_apply_policy: xs, y = self.policy(xs, y)

            batch_x[0][i] = left_image
            batch_x[1][i] = right_image
            
            batch_y[0][i,0] = disparity
            batch_y[1][i,0] = left_image.copy()
            batch_y[1][i,1] = right_image.copy()

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        # exit()

        return batch_x, batch_y
