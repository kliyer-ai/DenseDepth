import os, sys, glob, time, pathlib, argparse
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data
from model import create_model
from data import get_train_test_data
from callbacks import get_callbacks

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

from loss import point_wise_depth, edges, ssim

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='disparity_data.zip', type=str, help='Training dataset.')
parser.add_argument('--encoder', default='dense169', type=str, help='Encoder model architecture.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--name', type=str, default='wires', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')

args = parser.parse_args()

# Inform about multi-gpu training
# check with nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus == 1: 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

# Create the model
model = create_model( existing=args.checkpoint, encoder=args.encoder )

# Data loaders 
train_generator, test_generator = get_train_test_data( args.bs, data_zipfile=args.data)

# Training session details
runID = datetime.now().strftime('%d-%b-%H:%M') + '-m' + args.encoder + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

 # (optional steps)
if True:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+__file__, 'w') as training_script: training_script.write(training_script_content)

    # Generate model plot
    plot_model(model, to_file=runPath+'/model_plot.svg', show_shapes=True, show_layer_names=True)

    # Save model summary to file
    from contextlib import redirect_stdout
    with open(runPath+'/model_summary.txt', 'w') as f:
        with redirect_stdout(f): model.summary()

# Multi-gpu setup:
basemodel = model
if args.gpus > 1: 
    model = multi_gpu_model(model, gpus=args.gpus)
    print('multi GPU ready!')

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=depth_loss_function, optimizer=optimizer, metrics=[point_wise_depth, edges, ssim])

print('Ready for training!\n') 

# Callbacks
callbacks = get_callbacks(model, basemodel, train_generator, test_generator, runPath, load_test_data() if args.full else None)

# Start training
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True)

# Save the final trained model:
with open(runPath+'/model.yaml', 'w') as f:
        f.write(basemodel.to_yaml())
basemodel.save_weights(runPath + '/model_weights.h5', )