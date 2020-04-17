from keras.models import model_from_yaml, Model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, compute_errors, print_errors
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.transform import resize
from shape import get_shape_depth
from custom_objects import custom_objects

from data import get_data, MonoTestSequence, extract_zip
import argparse
import os

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--data', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flip = False
show_diff = True

with open(args.model+'.yaml', 'r') as f:
    yaml_string = f.read()
    model = model_from_yaml(yaml_string, custom_objects=custom_objects)
    model.load_weights(args.model+'_weights.h5')

print('\nModel loaded ({0}).'.format(args.model))
input_left = model.inputs[0]
disp_left = model.get_layer('disp_left').get_output_at(0),
model = Model(input=input_left, output=disp_left)

bs = 4

data = extract_zip(args.data)
test = list((row.split(',') for row in (data['test_data/data.csv']).decode("utf-8").split('\n') if len(row) > 0))

test_generator = MonoTestSequence(data, test, bs)
N = len(test_generator)


preds = []
gts = []

for i in range(N):
    xs, ys = test_generator[i]
    ys_pred = model.predict(xs, batch_size=bs)

    if flip:
        xs_flipped = np.flip(xs, axis=2)
        y_pred_hat = model.predict(xs_flipped, batch_size=bs)
        y_pred_hat = np.flip(y_pred_hat, axis=2)
        y_pred = (y_pred + y_pred_hat) / 2.0

    preds.append(ys_pred)
    gts.append(ys)

preds = np.stack(preds, axis=0)
gts = np.stack(gts, axis=0)

e = compute_errors(gts, preds, mask=False)

print_errors(e)