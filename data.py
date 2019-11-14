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

def get_train_test_data(batch_size, data_zipfile):
    data, train, test = get_data(batch_size, data_zipfile)

    train_generator = BasicAugmentRGBSequence(data, train, batch_size=batch_size)
    test_generator = BasicRGBSequence(data, test, batch_size=batch_size)

    return train_generator, test_generator

class BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, is_flip=True, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.50, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=0.0 if not is_erase else 0.50)
        self.batch_size = batch_size

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( get_shape_rgb(batch_size=self.batch_size) ), np.zeros( get_shape_depth(batch_size=self.batch_size, halved=True) )
        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(get_shape_rgb())/255,0,1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(get_shape_depth())/255,0,1)

            batch_x[i] = resize(x, get_shape_rgb()[1])
            batch_y[i] = resize(y, get_shape_depth(halved=True)[1])

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( get_shape_rgb(batch_size=self.batch_size) ), np.zeros( get_shape_depth(batch_size=self.batch_size, halved=True) )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(get_shape_rgb())/255,0,1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(get_shape_depth())/255,0,1).copy().astype(float)

            batch_x[i] = resize(x, get_shape_rgb()[1])
            batch_y[i] = resize(y, get_shape_depth(halved=True)[1])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

