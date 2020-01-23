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
from data import get_data, MonoTestSequence

parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--gt', default='', type=str, help='Input filename or folder.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with open(args.model+'.yaml', 'r') as f:
    yaml_string = f.read()
    model = model_from_yaml(yaml_string, custom_objects=custom_objects)
    model.load_weights(args.model+'_weights.h5')

print('\nModel loaded ({0}).'.format(args.model))
input_left = model.inputs[0]
disp_left = model.get_layer('disp_left').get_output_at(0),
model = Model(input=input_left, output=disp_left)

bs = 4

data = extract_zip(data_zipfile)
test = list((row.split(',') for row in (data['data/data.csv']).decode("utf-8").split('\n') if len(row) > 0))

test_generator = MonoTestSequence(data, test, bs)
N = len(test_generator)


predictions = []
testSetDepths = []

for i in range(N):
    xs_test, y_test = test_generator[i]
    y_pred = model.predict(xs_test, batch_size=bs)

    disps_test = y_test[0]
    disp_left_test = disps_test[:,0]

    disps_pred = y_pred[0]
    disp_left_pred = disps_pred[:,0]

    predictions.append(disp_left_pred)
    testSetDepths.append(disp_left_test)

predictions = np.stack(predictions, axis=0)
testSetDepths = np.stack(testSetDepths, axis=0)

e = compute_errors(testSetDepths, predictions)

if verbose:
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))