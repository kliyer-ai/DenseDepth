import numpy as np
from PIL import Image
from shape import get_shape_rgb, get_shape_depth
import keras.backend as K


# Support multiple RGBs, one RGB image, even grayscale 
def normalize_image_dims(image):
    if len(image.shape) < 3: image = np.stack((image,image,image), axis=2)
    if len(image.shape) < 4: image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def predict(model, inputs, batch_size=1):
    images = inputs[:2]
    num_disp = [inputs[-1]]
    if isinstance(images, list): 
        for i in range(len(images)):
            images[i] = normalize_image_dims(images[i])
    else: images = normalize_image_dims(images)

    predictions = model.predict(images + [np.array(num_disp)], batch_size=batch_size)

    return [np.clip(pred, 0.0, 1.0) for pred in predictions]

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []
    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)

def load_images(image_files):
    image_files.sort()
    print(image_files)
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open( file ), dtype=float) / 255, 0, 1)
        loaded_images.append(normalize_image_dims(x))
    return loaded_images # np.stack(loaded_images, axis=0)

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)
        
def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    
    all_images = []

    # iterate over batches
    for i in range(outputs.shape[0]):
        imgs = []
        
        if isinstance(inputs, np.ndarray):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)
        elif isinstance(inputs, list):
            for _input in inputs:
                x = to_multichannel(_input[i])
                x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
                imgs.append(x)
        
        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    
    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage =  display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage*255))
    im.save(filename)

def resize(img, resolution, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, resolution), preserve_range=True, mode='reflect', anti_aliasing=True )

def down_scale(img, factor):
    from skimage.transform import downscale_local_mean
    return downscale_local_mean(img, (2,2,1))

def load_test_data(test_data_zip_file, nr_inputs=1):
    print('Loading test data...', end='')
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    dataset = list((row.split(',') for row in (data['data/test.csv']).decode("utf-8").split('\n') if len(row) > 0))
    n = len(dataset)

    shape_rgb = get_shape_rgb(batch_size=n)
    shape_depth = get_shape_depth(batch_size=n, halved=True)

    batch_y = np.zeros(shape_depth)
    batches_x = []
    for _ in range(nr_inputs):
        batches_x.append(np.zeros(shape_rgb))

    from io import BytesIO
    for i in range(n):
        sample = dataset[i]

        for input_nr in range(nr_inputs):
            x = np.clip(np.asarray(Image.open( BytesIO(data[sample[input_nr]]) )).reshape(get_shape_rgb())/255,0,1)
            x = resize(x, get_shape_rgb()[1])
            batches_x[input_nr][i] = x

        y = np.clip(np.asarray(Image.open( BytesIO(data[sample[-2]]) )).reshape(get_shape_depth())/255,0,1)
        batch_y[i] = resize(y, get_shape_depth(halved=True)[0])

    print('Test data loaded.\n')
    return {'rgb':batches_x, 'depth':batch_y, 'crop':None}


def compute_errors(gt, pred):
    print(gt.shape)
    print(pred.shape)

    # masking to only compute loss over wires where disparity is defined 
    mask = gt > 0.0
    pred = pred[mask]
    gt = gt[mask]
    
    pred = np.clip(pred * 255, 1, 255)
    gt = np.clip(gt * 255, 1, 255) 

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    sqrt_rel = ((gt - pred) ** 2 / gt).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, sqrt_rel

def evaluate(model, test_generator, batch_size=4, verbose=False):
    N = len(test_generator)
    print('N: ', N)
    bs = batch_size

    predictions = []
    testSetDepths = []

    for i in range(N):
        xs_test, y_test = test_generator[i]
        y_pred = model.predict(xs_test, batch_size=4)

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

    return e
