import sys

from keras import applications
from keras.models import Model, load_model, Sequential
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, Lambda
from layers import BilinearUpSampling2D
from bilinear_sampler import generate_image_left, generate_image_right
import tensorflow as tf
from custom_objects import custom_objects

def get_decoders(models, num_disp, is_halffeatures=True):
        left_img = models[0].input
        right_img = models[1].input

        # Define upsampling layer
        def upproject(tensors, filters, name, concat_with):
            layer = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')
            tensors = list(map(lambda x: layer(x) , tensors))

            # Concatenate is only layer that exists separate for each input
            if concat_with == 'input':
                skips = [
                    models[0].get_layer('input_1').get_output_at(0),
                    models[1].get_layer('input_2').get_output_at(0)
                ]
            else:
                skips = [
                    models[0].get_layer(concat_with).get_output_at(0),
                    models[1].get_layer(concat_with).get_output_at(1)
                ]
            concat_left = Concatenate(name=name+'_concat_left')([tensors[0], skips[0]])
            concat_right = Concatenate(name=name+'_concat_right')([tensors[1], skips[1]])
            tensors = [concat_left, concat_right]

            layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')
            tensors = list(map(lambda x: layer(x), tensors))
            layer = LeakyReLU(alpha=0.2)
            tensors = list(map(lambda x: layer(x), tensors))
            layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')
            tensors = list(map(lambda x: layer(x), tensors))
            layer = LeakyReLU(alpha=0.2)
            tensors = list(map(lambda x: layer(x), tensors))
            return tensors

        # Starting point for decoder
        model_output_shape = models[0].layers[-1].get_output_at(0).shape

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(model_output_shape[-1]) // 2
        else:
            decode_filters = int(model_output_shape[-1]) 

         # Decoder Layers
        layer = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=model_output_shape, name='conv2')
        decoders = list(map(lambda x: layer(x), list(map(lambda m: m.output , models))))
        decoders = upproject(decoders, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoders = upproject(decoders, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoders = upproject(decoders, int(decode_filters/8), 'up3', concat_with='pool1')
        decoders = upproject(decoders, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if True: decoders = upproject(decoders, int(decode_filters/32), 'up5', concat_with='input')

        # Extract depths (final layer)
        left_disp = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='disp_left')(decoders[0])
        right_disp = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='disp_right')(decoders[1])
        right_disp = Lambda(lambda x: tf.reverse(x, axis=[2]), name='reverse_disp_right')(right_disp)  # 2 because of batch

        left_reconstruction = Lambda(lambda x: generate_image_left(x[0], x[1], x[2]), name='recon_left')([right_img, left_disp, num_disp])
        right_reconstruction = Lambda(lambda x: generate_image_right(x[0], x[1], x[2]), name='recon_right')([left_img, right_disp, num_disp])
        
        disparities = Lambda(lambda xs: tf.stack(xs, axis=1), name='disparities')([left_disp, right_disp])
        reconstructions = Lambda(lambda xs: tf.stack(xs, axis=1), name='reconstructions')([left_reconstruction, right_reconstruction])

        return disparities, reconstructions

def create_model(existing='', encoder='dense169'):
        
    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        # Encoder Layers
        if encoder=='dense201':
            left_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False) 
        elif encoder=='dense169':
            left_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False) 
        elif encoder=='dense121':
            left_model = applications.DenseNet121(input_shape=(None, None, 3), include_top=False) 

        # Layer freezing?
        for layer in left_model.layers: layer.trainable = True

        right_model = add_right_input(left_model)

        num_disp = Input(shape=(1,), dtype=tf.int16)

        print(left_model.input)
        print(right_model.input)
        print(num_disp)

        print('Base model loaded.')

        disparities, reconstructions = get_decoders([left_model, right_model], num_disp)
        print(disparities)
        print(reconstructions)

        # Create the model
        model = Model(inputs=[left_model.input, right_model.input, num_disp], outputs=[disparities, reconstructions])
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model

# modified from
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def add_right_input(model):

    _input = Input(shape=(None,None,3))

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if (layer.name not in network_dict['input_layers_of'][layer_name]):
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: _input})

    x = _input

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        if layer.name == 'zero_padding2d_1':
            new_layer = Lambda(lambda x : tf.reverse(x, [2]), name='reverse_input_right')
            new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            x = layer(x)
        else: 
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=_input, outputs=x)