########################################################################
#
# functions for plot images and confusion matrix
#
#
# Implemented in Python 3.6
#
########################################################################
########################################################################
from modules.mnist import MNIST
data = MNIST(data_dir="data/MNIST")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

num_classes = data.num_classes
img_shape = data.img_shape

########################################################################
#
# helper founction for ploting images
# function used to plot 9 images in a 3*3 grid,
# and writing the true and predicted classes below each image
#
########################################################################
# this function is called by function plot_example_error() below.
def plot_images(images, cls_true, cls_pred=None):
     
    assert len(images) == len(cls_true) == 9
    
    # create figure with 3*3 subplot
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # plot images
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        
        # show true and predicted classes
        if cls_pred is None:
            xlable = "True: {0}".format(cls_true[i])
        else:
            xlable = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        
        # show the classes as the label on the x-axis.
        ax.set_xlabel(xlable)
        
        # remove the ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    # plot images
    plt.show()

# helper-fuction to plot example errors
# plot the fist 9 images which not be correctly classified in the test set
def plot_example_error(cls_pred, correct):
    # this function is called by function print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for all images in the test-set
    print("    predicted classes:\t", cls_pred)
    #correct is a boolean array whether the predicted class is equal to the true class
    # for each iamge in the test-set
    print("    correct:\t\t", correct)
    
    incorrect = (correct == False)
    print("    incorrect:\t\t", incorrect)
    print("------------------------------------------------------------------------------------")
    
    # get the images from the test-set that have been incorrectly classied
    images  = data.x_test[incorrect]
    
    # get the predicted classes fo those images
    cls_pred = cls_pred[incorrect]
    
    # get the true classes for those images
    cls_true = data.y_test_cls[incorrect]
    
    # plot the first 9 images
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    print("------------------------------------------------------------------------------------")
########################################################################
#
# helper function to plot confusion matrix
#
########################################################################
def plot_confusion_matrix(cls_true, cls_pred):
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print("Confusion Matrix:\n", cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show
    print("------------------------------------------------------------------------------------")
########################################################################