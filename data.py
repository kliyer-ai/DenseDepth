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

    train_generator = BasicRGBSequence(data, train, batch_size, train=True)
    test_generator = BasicRGBSequence(data, test, batch_size)

    return train_generator, test_generator

class BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, train=False ,is_flip=True, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.50, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=0.0 if not is_erase else 0.50)
        self.batch_size = batch_size
        self.train = train

        from sklearn.utils import shuffle
        if self.train:
            self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        shape_rgb = get_shape_rgb(batch_size=self.batch_size)
        shape_depth = get_shape_depth(batch_size=self.batch_size, halved=True)
        nr_inputs = len(self.dataset[0]) - 2

        batch_y = np.zeros(shape_depth)
        batches_x = []
        for _ in range(nr_inputs):
            batches_x.append(np.zeros(shape_rgb))


        # Augmentation of RGB images
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)
            sample = self.dataset[index]

            xs = []
            for input_nr in range(nr_inputs):
                x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[input_nr]]) )).reshape(get_shape_rgb())/255,0,1)
                x = resize(x, get_shape_rgb()[1])
                xs.append(x)

            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[-2]]) )).reshape(get_shape_depth())/255,0,1).copy().astype(float)
            y = resize(y, get_shape_depth(halved=True)[1])

            if self.train and is_apply_policy: xs, y = self.policy(xs, y)

            for input_nr, x in enumerate(xs):
                batches_x[input_nr][i] = x
            batch_y[i] = y

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batches_x, batch_y
