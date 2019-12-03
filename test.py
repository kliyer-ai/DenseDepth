import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import model_from_yaml
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, compute_errors
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.transform import resize
from shape import get_shape_depth

def print_info(img):
    print('min', img.min())
    print('max', img.max())
    print('mean', img.mean())

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
# model = load_model(args.model, custom_objects=custom_objects, compile=False)
with open(args.model+'.yaml', 'r') as f:
    yaml_string = f.read()
    model = model_from_yaml(yaml_string, custom_objects=custom_objects)
    model.load_weights(args.model+'_weights.h5')

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )

# GT
base_path, _ = os.path.split(args.input)
gt = np.clip(np.asarray(Image.open( os.path.join(base_path, 'gt.png') ), dtype=float) / 255, 0, 1)
# downsample
gt = resize(gt, get_shape_depth(halved=True), preserve_range=True, mode='reflect', anti_aliasing=True )
# print_info(gt)

# Compute results
output = predict(model, inputs)
# print_info(output)

metrics = compute_errors(gt, np.reshape(output, get_shape_depth(halved=True)))
print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
viz = display_images(output.copy(), inputs.copy(), is_colormap=False)
split_path = os.path.normpath(args.input).split('/')
if len(split_path) < 2: path = 'test.png'
else: path = split_path[-2] + '_pred.png'

print('saved at', path)
# cv.imwrite(path, np.uint8(viz*255))

plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig(path)
plt.show()
