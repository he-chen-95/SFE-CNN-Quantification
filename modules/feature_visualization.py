########################################################################
#
# Functions for build basic cnn modules.
#
# Implemented in Python 3.6
#
########################################################################
import tensorflow as tf
import numpy as np

from modules.plot_functions import plot_images_10

from modules.mnist import MNIST
data = MNIST(data_dir="data/MNIST")

img_shape = data.img_shape
########################################################################
#
# Functions for getting the names of covolutional layers
#
########################################################################
def get_conv_layer_names():
    graph = tf.get_default_graph()
    
    # create a list of names for the operations in the graph
    # for this CNN model where the operator-type is 'Conv2D'
    names = [op.name for op in graph.get_operations() if op.type=='Conv2D']
    
    return names

########################################################################
#
# Functions for finding the input image
#
# this function finds the input image that maximizes a given feature
# in the network. it essentially just performs optimization with 
# gradient ascent. The image is intialized with small random values and 
# is then iteratively updated using the gradient for the given feature
# with regard to the image
########################################################################
def optimize_image(tf_session, x_image, y_pred, logits, conv_names, 
                   conv_id=None, feature=0,
                   num_iterations=30, show_progress=True):
    """
    Find the image that maximizes the feature given by 
    the conv_id and feature number.
    Parameters:
    conv_id: 
    Integer identifying the convolutional layer to mximize. It is an index
    into conv_names. If None then use the last fully-connected layer before 
    the softmax output.
    feature: 
    Index into the layer for the feature to maximize.
    num_iteration: 
    number of optimisation iterations to performe.
    show_progress: 
    Boolean whether to show the progress.
    """
    # create the loss-function 
    if conv_id is None:
        # if we want to maximze a feature on the last layer,
        # then we use the FC layer prior to the softmax-classifier
        # and must be an integer between 0 and 15 for the 1st conv layer
        # (5*5*1*16). then 0-35 for the 2nd conv layer(5*5*16*36).
        # the loss function is just the value of that feature
        loss = tf.reduce_mean(logits[:, feature])
    else:
        # if instead we want maximize a feature of a convolutional layer
        # inside the neural network.
        
        # get the name of the convolutional oprator
        conv_name = conv_names[conv_id]
        
        # get the default TF graph.
        graph = tf.get_default_graph()
        
        # get a reference to the tensor that is output by the operator.
        # Note that ":0" is added to the name for this.
        tensor = graph.get_tensor_by_name(conv_name + ":0")
        
        # the loss-function is the average of all the tensor-values
        # for the given feature. This ensure that we generate the 
        # whole input image. you can try and modify this so it only
        # uses a part of the tensor.
        loss = tf.reduce_mean(tensor[:, :, :, feature])
    
    # get the gradient for the loss function with regard to the input image
    # this creates a mathematical function for calcuting the gradient 
    gradient = tf.gradients(loss, x_image)
    
    # generate a random image of the same size as the raw input.
    # each pixel is a small random value between 0.45 and 0.55,
    # which is the middle of the valide range between 0 and 1.
    image = 0.1 * np.random.uniform(size = img_shape) + 0.45
    
    # performe a number of optimisation iterations to find the image
    # that maximizes the loss-function
    for i in range(num_iterations):
        # reshape the array so it is a 4-rank tensor.
        img_reshaped = image[np.newaxis, :, :, np.newaxis]
        
        # create a feed-dict for inputting the image to the graph
        feed_dict = {x_image: img_reshaped}
        
        # calculate the predicted class-scores, as well as 
        # the gradient and the loss-value.
        pred, grad, loss_value = tf_session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)
        
        # squeeze the dimensionality for the gradien-array.
        grad = np.array(grad).squeeze()
        
        # the gradient now tells us how much we need to change
        # the input image in order to maximize the given feature
        
        # calculate the step-size for updating the image.
        # this step-size was found to give fast convergence
        # the addition og 1e-8 is to prevent to div-by-zero.
        step_size = 1.0 / (grad.std() + 1e-8)
        
        # update the image by adding the sacled gradient
        # this called gradient descent.
        image += step_size * grad
        
        # Ensure all pixel-values in the image are between  0 and 1.
        image = np.clip(image, 0.0, 1.0)
        
        if show_progress:
            print("Iteration:", i)
            
            # convert the predicted class-score to a 1-dimension array.
            pred = np.squeeze(pred)
            
            # the predicted class for the CNN model
            pred_cls = np.argmax(pred)
            
            # the score(probability) for the predicted class.
            cls_score = pred[pred_cls]
            
            # print the predicted score etc.
            msg = "Predicted class: {0}, score: {1:7.2%}"
            print(msg.format(pred_cls, cls_score))
            
            # print statistics for the gradient.
            msg = "Gradient min: {0:9.6f}, max: {1:9.6f}, step_size:{2:9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))
            
            # print the loss value.
            print("Loss:", loss_value)
            
            # newline
            print()
    return image.squeeze()

########################################################################
#
# Functions for finding the input image
# The next function finds the image that maximize the first 10 feature 
# of a layer, by calling the above function 10 times.
#
########################################################################
def optimize_images(tf_session, x_image, y_pred, logits, 
                    conv_names, conv_id=None, num_iterations=30):
    """
    Find 10 images that maximize the 10 first features in the layer
    given by the conv_id
    
    Parameters:
    conv_id:
    Integer identifying the covolutional layer to maximize. It is 
    an index into conv_names.
    num_iterations: Number of optimisation iterations to performe.
    """
    
    # which layer are we using?
    if conv_id is None:
        print("Final FC layer before softmax.")
    else:
        print("Layer:", conv_names[conv_id])
    
    # initialize the array og images.
    images = []
    
    # for each feature, do the following.
    for feature in range(0, 10):
        print("Optimizing image for feature No.", feature)
        
        # find the image that maximizes the given feature for
        # the network layer identified by conv_id(or None)
        image = optimize_image(tf_session=tf_session,
                               x_image = x_image,
                               y_pred = y_pred,
                               logits = logits,
                               conv_names=conv_names,
                               conv_id=conv_id, 
                               feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)
        
        # squeeze the dimension of the array
        image = image.squeeze()
        
        # append to the list of images.
        images.append(image)
        
    # convert to numpy array so we can index all dimension easily.
    images = np.array(images)
    
    # plot the images.
    plot_images_10(images=images)
