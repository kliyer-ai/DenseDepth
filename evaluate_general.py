from keras.models import model_from_yaml, Model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, compute_errors, down_scale, print_errors
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.transform import resize
from shape import get_shape_depth
from custom_objects import custom_objects

import argparse
import os

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='', type=str, help='Trained Keras model file.')
parser.add_argument('--name', default='', type=str, help='extra name')
args = parser.parse_args()

data_path = [
    ('data/folder1/test0.png', 'data/folder1/gt0.png'),
    ('data/folder2_cropped/test0.png', 'data/folder2_cropped/gt0.png'),
    ('data/folder6_cropped/test0.png', 'data/folder6_cropped/gt0.png'),
    # ('data/folder1/test/230_img0.png', 'data/folder1/test/230_depth0.png'),
    # ('data/folder1/test/755_img0.png', 'data/folder1/test/755_depth0.png'),
    # ('data/folder2_cropped/test/105_img0.png', 'data/folder2_cropped/test/105_depth0.png'),
    # ('data/folder6_cropped/test/95_img0.png', 'data/folder6_cropped/test/95_depth0.png')
    # ('data/folder1/test/595_img0.png', 'data/folder1/test/595_depth0.png'),
    # ('data/folder2_cropped/test/55_img0.png', 'data/folder2_cropped/test/55_depth0.png')
]

# scales = np.array([144, 64, 64, 144, 144, 64, 64])
scales = np.array([144, 64, 64])
# scales = np.array([64])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flip = True
flip_factor = 0.05
show_diff = True

with open(args.model+'.yaml', 'r') as f:
    yaml_string = f.read()
    model = model_from_yaml(yaml_string, custom_objects=custom_objects)
    model.load_weights(args.model+'_weights.h5')

print('\nModel loaded ({0}).'.format(args.model))
input_left = model.inputs[0]
disp_left = model.get_layer('disp_left').get_output_at(0),
model = Model(input=input_left, output=disp_left)

xs = []
ys = []

for (x_path, y_path) in data_path:
    x = np.clip(np.asarray(Image.open( x_path ), dtype=float) / 255, 0, 1)
    y = np.clip(np.asarray(Image.open( y_path ), dtype=float) / 255, 0, 1)
    y = np.expand_dims(y, axis=-1)
    y = down_scale(y, 2)
    xs.append(x)
    ys.append(y)

xs = np.stack(xs)
ys = np.stack(ys)

bs = xs.shape[0]

y_pred = model.predict(xs, batch_size=bs)



if flip:
    width = y_pred.shape[2]
    left_start = int(width * flip_factor)
    right_start = int(width - width * flip_factor)
    xs_flipped = np.flip(xs, axis=2)
    y_pred_hat = model.predict(xs_flipped, batch_size=bs)
    y_pred_hat = np.flip(y_pred_hat, axis=2)
    y_pred_merged = (y_pred + y_pred_hat) / 2.0
    y_pred_merged[:, :, :left_start, :] = y_pred_hat[:, :, :left_start, :]
    y_pred_merged[:, :, right_start:, :] = y_pred[:, :, right_start:, :]
    y_pred = y_pred_merged


e = compute_errors(ys, y_pred, scales)

print(args.name)
print_errors(e)

diff = None
if show_diff: diff = np.abs(y_pred - ys)

viz = display_images(y_pred, xs, gt=ys, diff=diff, is_colormap=False)
plt.figure(figsize=(10,20)) # 
plt.imshow(viz)
plt.axis('off')
plt.savefig(args.name + '_results')
plt.show()


# if show_diff:
#     y_pred = np.abs(y_pred - ys)
#     viz = display_images(y_pred, xs, gt=None, is_colormap=False)
#     plt.figure(figsize=(10,20))
#     plt.imshow(viz)
#     plt.savefig(args.name + '_test_results_diff')
#     plt.show()