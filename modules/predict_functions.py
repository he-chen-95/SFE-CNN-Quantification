########################################################################
#
# helper functions for classes prediction
#
#
# Implemented in Python 3.6
#
########################################################################
from modules.mnist import MNIST
data = MNIST(data_dir="data/MNIST")

import tensorflow as tf
import numpy as np

########################################################################
#
# fonction to perform optimization iterations
#
########################################################################

import time
from datetime import timedelta

# best accuracy seen so far
best_validation_accuracy = 0.0

# iteration-number for last improvement to validation_accuracy
last_improvement = 0

# stop optimization if have no improvement afters 1000 iterations
require_improvement = 1000

# calculate the predicted class for the test-set
total_iterations = 0
# train batch size = 0 par default
# dropout probability = 0.75 par default
# tf_session for runing optimization
# tf_saver for saving weights file
# merged_summary for visualizing loss, accuracy on tensorboard
def optimize(tf_session,
             tf_saver,
             tensorboard_summary,
             tensorboard_writer,
             model_path,
             x_ph,
             y_true_ph,
             keep_prob_ph,
             y_pred_cls_tensor,
             optimizer,
             display_step,
             num_iterations, 
             loss_opt,
             accuracy,
             train_batch_size=64, 
             dropout_prob=0.75):
    
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time = time.time()
    
    # op to write logs to tensorboard
    #summary_writer = tf.summary.FileWriter(logs_path, 
    #                                       graph=tf.get_default_graph())
    
    for step in range(total_iterations, 
                   total_iterations + num_iterations):
                
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)
        
        # at training time, dropout probability != 1, prevent overfitting.
        feed_dict_train_dropout = {x_ph: x_batch,
                                   y_true_ph: y_true_batch,
                                   keep_prob_ph: dropout_prob}
        
        # run the merged_summary
        result = tf_session.run(tensorboard_summary, feed_dict=feed_dict_train_dropout)
        # write it to disk
        tensorboard_writer.add_summary(result, step)
        
        # run the training step
        tf_session.run(optimizer, feed_dict=feed_dict_train_dropout)
        
        # run optimization op(backprop), cost op(to get loss value),
        # and summary nodes
        #_, c, summary = tf.session.run([apply_grads, cost, merged_summary],
        #                               feed_dict=feed_dict_train_dropout)
        
        # write logs at every iteration
        #summary_writer.add_summary(summary, step)
        
        # at test time, dropout probability = 1, all connections always present 
        feed_dict_dropout = {x_ph: x_batch,
                             y_true_ph: y_true_batch,
                             keep_prob_ph: 1.0}
        
        
        if ((step % display_step == 0) or (step == 0)):
        # if ((step % display_step == 0) or (step == (num_iterations - 1))):
            
            # occationally report accuracy
            # acc_train = tf_session.run(accuracy, feed_dict=feed_dict_train_dropout)
            loss, acc_train = tf_session.run([loss_opt, accuracy], feed_dict=feed_dict_dropout)

            acc_validation, _ = validation_accuracy(tf_session=tf_session,
                                                    x_ph=x_ph,
                                                    y_true_ph=y_true_ph,
                                                    keep_prob_ph=keep_prob_ph,
                                                    y_pred_cls_tensor=y_pred_cls_tensor)
            
            if acc_validation > best_validation_accuracy:
                # update the best-known validation accuracy
                best_validation_accuracy = acc_validation
                
                # set the iteration fot the last improvement to current
                last_improvement = total_iterations
                
                # save all variables of the tensorflow graph of file
                print("model_path:", model_path)
                load_path = tf_saver.save(sess=tf_session, save_path=model_path, global_step=step+1)
                print("Model saved in file: %s" %load_path)
                # s string to be printed below, show improvement found
                improved_str = '*'
            else:
                improved_str = ''
                
            msg = "Optimization Iteration: {0:>6}, Minibatch Loss:{1:.4f}, Training-Batch Accuracy: {2:>6.1%}, Validation Accuracy:{3:>6.1%} {4}"
            print(msg.format(step + 1,loss ,acc_train, acc_validation, improved_str))
                        
            #print("Step " + str(step) + ", Minibatch Loss= " + \
            #      "{:.4f}".format(loss) + ", Training-Batch Accuracy= " + \
            #      "{:.3f}".format(acc_train) + ", Validation Accuracy=" + \
            #      "{:.3f}".format(acc_validation) + "{3}".format(improved_str)
            
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization")
            break
    
    total_iterations += num_iterations
            
    end_time = time.time()
            
    time_dif = end_time - start_time
            
    print("Time usage:"+str(timedelta(seconds=int(round(time_dif)))))
    
    return load_path
########################################################################
#
# helper functions for calculating classifications
#
########################################################################
def predict_cls(tf_session, 
                x_ph, 
                y_true_ph, 
                keep_prob_ph, 
                y_pred_cls_tensor, 
                y_true_cls, 
                images, 
                labels, 
                batch_size=256):
    
    # print("x_ph:", x_ph)
    # print("y_true_ph", y_true_ph)
    # batch_size=256
    # Split the data-set in batches of this size to limit RAM usage.
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    # 1D array with length of num_test
    # allocate this array for predicted classes which
    # will be calculated in batches and filled into this array
    # ex. [0, 0, 0, 0, ... 0, 0]
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        # at test time, keep dropout probability = 1
        # Tensor names must be of the form "<op_name>:<output_index>".
        feed_dict = {x_ph: images[i:j, :],
                     y_true_ph: labels[i:j, :],
                     keep_prob_ph: 1.0}
        
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = tf_session.run(y_pred_cls_tensor, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    cls_true = y_true_cls
    correct = (cls_true == cls_pred)
    # correct = tf.equal(cls_true, cls_pred)
    # when we use tf.equal, it return a tensor
    # for test set, it returns
    # correct: Tensor("Equal_1:0", shape=(10000,), dtype=bool)
    # for the validation set, it returns
    # correct: Tensor("Equal_2:0", shape=(5000,), dtype=bool)
    # in cls_accuracy(), i get the following error
    # AttributeError: 'Tensor' object has no attribute 'sum'
    
    #print("    predicted classes:\t", cls_pred)
    #print("    true classes:\t", cls_true)
    #print("    correct:\t\t", correct)
    #print("------------------------------------------")
    return correct, cls_pred

# called by print_test_accuracy()
def predict_cls_test(tf_session, x_ph, y_true_ph, keep_prob_ph, y_pred_cls_tensor):
    print("predicted classes for test-set:")
    return predict_cls(tf_session = tf_session,
                       x_ph = x_ph, 
                       y_true_ph = y_true_ph,
                       keep_prob_ph = keep_prob_ph,
                       y_pred_cls_tensor = y_pred_cls_tensor,
                       y_true_cls = data.y_test_cls,
                       images = data.x_test,
                       labels = data.y_test)

# calculate the predicted class for the validation-set
# called by validation_accuracy()
def predict_cls_validation(tf_session, x_ph, y_true_ph, keep_prob_ph, y_pred_cls_tensor):
    print("predicted classes for validation-set:")
    return predict_cls(tf_session = tf_session, 
                       x_ph = x_ph, 
                       y_true_ph = y_true_ph,
                       keep_prob_ph = keep_prob_ph,
                       y_pred_cls_tensor = y_pred_cls_tensor,
                       y_true_cls = data.y_val_cls,
                       images = data.x_val,
                       labels = data.y_val)
########################################################################
#
# functions for the classification accuracy
#
########################################################################
# called by validation_accuracy()
# this function calcute the classification accuracy by giving a boolean
# array whether each images was correctly classified
# ex. cls_accuracy([True, False, True, False, False]) = 2/5 = 0.4
def cls_accuracy(correct):
    # calcute the number of correctly classified images.
    # when summing a boolean array, False means 0, True means 1.
    # print("correct in cls_accuracy:", correct)
    correct_sum = correct.sum()
       
    # classification accuracy is the number of correctly classified images
    # divided by the total number of images in the test-set
    acc = float(correct_sum) / len(correct)
    
    return acc, correct_sum

# called by optimize()
# calcute the classification accuracy on validation-set
def validation_accuracy(tf_session, x_ph, y_true_ph, keep_prob_ph, y_pred_cls_tensor):
    correct, _ = predict_cls_validation(tf_session=tf_session,
                                        x_ph=x_ph,
                                        y_true_ph=y_true_ph,
                                        keep_prob_ph=keep_prob_ph,
                                        y_pred_cls_tensor=y_pred_cls_tensor)
    
    return cls_accuracy(correct)


########################################################################
#
# fonction to perform optimization iterations
#
########################################################################    
from modules.plot_functions import plot_confusion_matrix
from modules.plot_functions import plot_example_error
# from modules.predict_functions import cls_accuracy
# from modules.predict_functions import predict_cls_test

def print_test_accuracy(tf_session,
                        x_ph,
                        y_true_ph,
                        keep_prob_ph,
                        y_pred_cls_tensor,
                        show_example_error=False,
                        show_confusion_matrix=False):

    # get the true labels for tset set
    cls_true_test = data.y_test_cls
    
    correct_test, cls_pred_test = predict_cls_test(tf_session=tf_session,
                                                   x_ph=x_ph,
                                                   y_true_ph=y_true_ph,
                                                   keep_prob_ph=keep_prob_ph,
                                                   y_pred_cls_tensor=y_pred_cls_tensor)
    
    # print("correct_test in print_test_accuracy:", correct_test)
    acc_test, num_correct_test = cls_accuracy(correct_test)
    
    num_images_test =  len(correct_test)
    
    msg = "Accuracy on Test-set:{0:.1%} ({1} / {2})"
    print(msg.format(acc_test, num_correct_test, num_images_test))
    
    if show_example_error:
        print("Example errors for test-set:")
        plot_example_error(cls_pred=cls_pred_test, correct=correct_test)
        
    if show_confusion_matrix:
        print("Confusion Matrix for test-set :")
        plot_confusion_matrix(cls_true=cls_true_test, cls_pred=cls_pred_test)
