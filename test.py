import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import model_from_yaml, Model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, compute_errors
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.transform import resize
from shape import get_shape_depth
from custom_objects import custom_objects

def print_info(img):
    print('min', img.min())
    print('max', img.max())
    print('mean', img.mean())

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--gt', default='', type=str, help='Input filename or folder.')
parser.add_argument('--name', default='', type=str, help='name as prefix')
parser.add_argument('--folder', default='', type=str, help='folder name')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print('Loading model...')

# Load model into GPU / CPU
with open(args.model+'.yaml', 'r') as f:
    yaml_string = f.read()
    model = model_from_yaml(yaml_string, custom_objects=custom_objects)
    model.load_weights(args.model+'_weights.h5')

print('\nModel loaded ({0}).'.format(args.model))
input_left = model.inputs[0]
disp_left = model.get_layer('disp_left').get_output_at(0),
simple_model = Model(input=input_left, output=disp_left)

# Input images
inputs = load_images( glob.glob(args.input) )
inputs.append(np.array([1]))

# GT
base_path, _ = os.path.split(args.input)
# gt = np.clip(np.asarray(Image.open( os.path.join(base_path, 'gt0.png') ), dtype=float) / 255, 0, 1)
gt = np.clip(np.asarray(Image.open( os.path.join(base_path, args.gt) ), dtype=float) / 255, 0, 1)
print(gt.shape)
# downsample
is_halved=True
gt = resize(gt, get_shape_depth(halved=is_halved), preserve_range=True, mode='reflect', anti_aliasing=True )
# print_info(gt)

# Compute results
disps, recons = predict(model, inputs)
left_disp = disps[:,0]
print_info(left_disp)
print(left_disp.shape)

metrics = compute_errors(np.reshape(gt, get_shape_depth(halved=is_halved)), np.reshape(left_disp, get_shape_depth(halved=is_halved)))
print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

print(inputs[0].shape)
print(inputs[1].shape)
print(inputs[2].shape)

# Display results
viz = display_images(left_disp.copy(), inputs[:1], gt=[gt], is_colormap=False)
# viz = display_images(left_disp.copy(), inputs[:1], is_colormap=False)
split_path = os.path.normpath(args.input).split('/')
if len(split_path) < 2: path = 'test.png'
else: path = split_path[-2] + '_pred_' + args.name + '.png'

if args.folder != '' and not os.path.isdir(args.folder):
    os.mkdir(args.folder)

path = args.folder + '/' + path

with open(path + '.txt', 'w') as f:
    f.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    f.write('\n')
    f.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))

print('saved at', path)
# cv.imwrite(path, np.uint8(viz*255))

plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig(path)
plt.show()
