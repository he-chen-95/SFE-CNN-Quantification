########################################################################
#
# Functions for build basic cnn modules.
#
# Implemented in Python 3.6
#
########################################################################

import tensorflow as tf

########################################################################

def new_conv_layer(input, 
                   layer_name,
                   weights,
                   biases,
                   use_pooling=True,
                   k=2, stride=2):
    
    # layer_name = 'conv_layer%s_'%n_layer
    # add namenode for tensorboard
    with tf.name_scope(layer_name):
        # tf.summary.histogram(layer_name+"/weights", weights)
        # INFO:tensorflow:Summary name /weights is illegal; using weights instead.
        layer = tf.nn.conv2d(input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv2d')
        #print("output of conv2d:", conv)
        
        layer = tf.nn.bias_add(layer, biases, name='w_plus_b')
        # layer = layer + biases
        # layer += biases
        #print("bias in conv:", biases)
        #print("output after w + b in conv:", layer)
        
        # in many papers, people use conv->pooling->non_linearity(activation_function)
        # MaxPool(ReLU(Conv(M))) == ReLu(MaxPool(Conv(M)))
        # if we use pooling->non-linearity, for pooling layer of size k, 
        # it uses k^2 times less calls to activation_function.
        # value: 4D tensor, input feature map
        # ksize: the size of pooling window for each dimension, 
        # [ksize_batch=1, ksize_height, ksize_width, ksize_channel=1].
        # strides: the stride of the sliding window for each dimension, for X/Y direction,
        # [stride_batch=1, stride_X, stride_Y, stride_channel=1]
        # padding: 'VALID' or 'SAME'  
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, k, k, 1],
                                   strides=[1, stride, stride, 1],
                                   padding='SAME',
                                   name='max_pooling')
                
        layer = tf.nn.relu(layer, name='relu')
        
        # add histogram to tensorboard in histograme bar
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("outputs", layer)
        return layer

def flatten_layer(layer, layer_name="flatten_layer"):
    with tf.name_scope(layer_name):
        # the shape of input is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]
        # shape=(?, 7, 7, 36)
        layer_shape = layer.get_shape()
    
        # num_features is: img_height * img_width * num_channels=7*7*36=1764
        # num_features = weights_dict['w_fc_1'].get_shape.as_list()[0]
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # the shape of flattened layer is now:
        # [num_images, img_height * img_width * num_channerls]
        # shape=(?, 1764)
        layer_flat = tf.reshape(layer, [-1, num_features], name='flatten')
        return layer_flat

def new_fc_layer(input,
                 layer_name,
                 weights,
                 biases,
                 use_relu=True):
    with tf.name_scope(layer_name):
        # layer = tf.matmul(input, weights) + biases
        layer = tf.add(tf.matmul(input, weights), biases, name='w_plus_b')
        if use_relu:
            layer = tf.nn.relu(layer, name='relu')
            
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)            
        tf.summary.histogram("outputs", layer)
        return layer

def new_dropout_layer(input,
                      layer_name,
                      weights,
                      biases,
                      dropout):
    with tf.name_scope(layer_name):
        layer = tf.nn.dropout(x=input, keep_prob=dropout, name='dropout')
        # layer = tf.matmul(layer, weights) + biases
        layer = tf.add(tf.matmul(layer, weights), biases, name='w_plus_b')  
        
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("outputs", layer)
        return layer
