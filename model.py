import sys

from keras import applications
from keras.models import Model, load_model, Sequential
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, Lambda
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from bilinear_sampler import bilinear_sampler_1d_h

def generate_image_left(img, disp):
    return bilinear_sampler_1d_h(img, -disp)

def generate_image_right(img, disp):
    return bilinear_sampler_1d_h(img, disp)

def create_model(existing='', encoder='dense169', is_halffeatures=True, nr_inputs=1):
        
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

        right_model = add_input(left_model)

        print('Base model loaded.')

        left_input_tensor = left_model.input
        right_input_tensor = right_model.input

        # Starting point for decoder
        left_model_output_shape = left_model.layers[-1].get_output_at(0).shape

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(left_model_output_shape[-1]) // 2
        else:
            decode_filters = int(left_model_output_shape[-1]) 

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, left_model.get_layer(concat_with).get_output_at(0)]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=left_model_output_shape, name='conv2')(left_model.output)

        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if True: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        disp_left = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='disp_left')(decoder)
        left_reconstruction = Lambda(lambda x: generate_image_left(right_input_tensor, x))(disp_left)

        # Create the model
        model = Model(inputs=[left_model.input, right_model.input], outputs=[disp_left, left_reconstruction])
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model


def add_input(model):

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

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=_input, outputs=x)