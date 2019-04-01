import tensorflow as tf

print("tensorflow version:", tf.__version__)

########################################################################
# load dataset
########################################################################
from modules.mnist import MNIST

data = MNIST(data_dir="data/MNIST")

print("Size of:")
print("    -Training-set:\t\t{}".format(data.num_train))
print("    -Validation-set:\t\t{}".format(data.num_val))
print("    -Test-set:\t\t\t{}".format(data.num_test))

# mnist images width and height: 28
img_size = data.img_size
# mnist data input (image shape: 28 * 28 = 784)
img_shape = data.img_shape
img_size_flat = data.img_size_flat
# mnist gray scale images with 1 channel
num_channels = data.num_channels
# mnist total classes(0-9 digits)
num_classes = data.num_classes

print("\ndata set information:")
print("    -img_size:\t\t\t", img_size)
print("    -img_shape:\t\t\t", img_shape)
print("    -num_channels:\t\t", num_channels)
print("    -img_size_flat:\t\t", img_size_flat)
print("    -num_classes:\t\t", num_classes)
########################################################################
# Training Parameters
########################################################################
learning_rate_1 = 1e-3
display_step = 20

train_batch_size_1 = 64
train_batch_size_2 = 128

# Dropout probability to keep units
dropout_prob_1 = 0.75

# L2 正则项系数
BETA = 0.01
########################################################################
# network parameters
########################################################################
# convolutional layer 1
size_conv_kernel_1 = 5
num_conv_kernel_1 = 16

# convolutional layer 2
size_conv_kernel_2 = 5
num_conv_kernel_2 = 36

# fully connected layer 1
# we do the pooling operations 2 times
num_pooling = 2
# fully connected layer, 7*7*36=1764 input size
size_input_fc_1 = int((img_size / (2 * num_pooling)) * (img_size / (2 * num_pooling)) * (num_conv_kernel_2))
# size of fc layer 1
size_fc_1 = 128

# fully connected layer 2 as output layer
size_output = num_classes
########################################################################
# layer definition
########################################################################
shape_dict = {
    # shape of weights

    # 5*5 conv, 1 input channel, 16 output channel.
    's_weights_conv_1': [size_conv_kernel_1, size_conv_kernel_1, num_channels, num_conv_kernel_1],
    # 5*5 conv, 16 input channel, 32 output channel.
    's_weights_conv_2': [size_conv_kernel_2, size_conv_kernel_2, num_conv_kernel_1, num_conv_kernel_2],
    # faltten layer, convolutional layer output 4D tensors. 
    # we now wish to use these as input in a fully connected network, 
    # which requires for the tensors to be reshaped or flattened to 2D tensors.
    # fully connected layer, 7*7*36=1764 inputs, 128 outputs
    's_weights_fc_1': [size_input_fc_1, size_fc_1],
    # fully connected layer, 128 inputs, 10 outputs(class prediction)
    's_weights_output_layer': [size_fc_1, num_classes],

    # shape of biases

    's_biases_conv_1': [num_conv_kernel_1],
    's_biases_conv_2': [num_conv_kernel_2],
    's_biases_fc_1': [size_fc_1],
    's_biases_output_layer': [size_output]
}

weights_dict = {

    # we add the name atrribute for tf.name_scope(). you find it in cnn_modules.py
    'w_conv_1': tf.Variable(tf.truncated_normal(shape_dict['s_weights_conv_1'], stddev=0.05), name="w_conv_1"),
    # tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtype.float32, seed=None, name=None)
    # --> a normal distribution, mean=0.0, standard deviation=0.1.
    # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtype.float32, seed=None, name=None)
    # --> a normal distribution, values whose magnitude is more than 2 standard deviations from the mean
    # are dropped and re-picked
    # other type include: tf.random_uniform, tf.random_shuffle, tf.random_crop, tf.multinomial,tf.random_gamme.
    'w_conv_2': tf.Variable(tf.truncated_normal(shape_dict['s_weights_conv_2'], stddev=0.05), name="w_conv_2"),

    'w_fc_1': tf.Variable(tf.truncated_normal(shape_dict['s_weights_fc_1'], stddev=0.05), name="w_fc_1"),

    'w_output_layer': tf.Variable(tf.truncated_normal(shape_dict['s_weights_output_layer'], stddev=0.05),
                                  name="w_output_layer")
}

biases_dict = {
    'b_conv_1': tf.Variable(tf.constant(0.05, shape=shape_dict['s_biases_conv_1']), name="b_conv_1"),
    # tf.constant(value, dtype=None, shape=None, name="Const", verify_shape=False)
    # --> a constant tensor.
    # tf.zeros(shape, dtype=tf.dtype.float32, name=None)
    # --> returns a tensor of 'dtype' with shape 'shape' and all elements set to 0.

    'b_conv_2': tf.Variable(tf.constant(0.05, shape=shape_dict['s_biases_conv_2']), name="b_conv_2"),

    'b_fc_1': tf.Variable(tf.constant(0.05, shape=shape_dict['s_biases_fc_1']), name="b_fc_1"),

    'b_output_layer': tf.Variable(tf.constant(0.05, shape=shape_dict['s_biases_output_layer']), name="b_output_layer")
}

# print(w_conv_1) function print the dtype, shape etc. information for this tensor
# print a tensor, please use tf.Session(w_conv_1)
print("w_conv_1:", weights_dict['w_conv_1'])
print("w_conv_2:", weights_dict['w_conv_2'])
print("w_fc_1:", weights_dict['w_fc_1'])
print("w_output_layer:", weights_dict['w_output_layer'])
print("b_conv_1:", biases_dict['b_conv_1'])
print("b_conv_2:", biases_dict['b_conv_2'])
print("b_fc_1:", biases_dict['b_fc_1'])
print("b_output_layer:", biases_dict['b_output_layer'])

########################################################################
# helper-function to create your CNN model
########################################################################
from modules.cnn_modules import new_conv_layer
from modules.cnn_modules import flatten_layer
from modules.cnn_modules import new_fc_layer
from modules.cnn_modules import new_dropout_layer


def conv_net(input, weights, biases, dropout_prob):
    # convolutional layer 1
    # name="layer_conv_1" means that we use it as the name node in tensorboard.
    layer_conv_1 = \
        new_conv_layer(input=x_image,
                       layer_name="layer_conv_1",
                       weights=weights_dict['w_conv_1'],
                       biases=biases_dict['b_conv_1'],
                       use_pooling=True)
    print("layer_conv_1:", layer_conv_1)

    # convolutional layer 2
    layer_conv_2 = \
        new_conv_layer(input=layer_conv_1,
                       layer_name="layer_conv_2",
                       weights=weights_dict['w_conv_2'],
                       biases=biases_dict['b_conv_2'],
                       use_pooling=True)
    print("layer_conv_2:", layer_conv_2)

    # flatten layer
    layer_flat = flatten_layer(layer_conv_2, layer_name="flatten_layer")
    print("layer_flat:", layer_flat)

    # fully-connected layer 1
    layer_fc_1 = new_fc_layer(input=layer_flat,
                              layer_name="layer_fc_1",
                              weights=weights_dict['w_fc_1'],
                              biases=biases_dict['b_fc_1'],
                              use_relu=True)
    print("layer_fc_1:", layer_fc_1)

    # fully-connected layer 2, output layer. relu is used,
    # so we can learn non-linear relations
    # layer_fc_2 = new_fc_layer(input=layer_fc_1,
    #                          layer_name="layer_fc_2",
    #                          weights=weights_dict['w_output_layer'],
    #                          biases=biases_dict['b_output_layer'],
    #                          use_relu=False)
    # print("layer_fc2:", layer_fc_2)

    # apply dropout layer as output layer
    layer_dropout = new_dropout_layer(input=layer_fc_1,
                                      layer_name="layer_dropout",
                                      weights=weights_dict['w_output_layer'],
                                      biases=biases_dict['b_output_layer'],
                                      dropout=dropout_prob)
    print("layer_dropout:", layer_dropout)

    layer_output = layer_dropout

    return layer_output


########################################################################
# feed-forword step
########################################################################
# Setup placeholders, add name nodes for tensorboard
# placeholder作为输入节点， Variable作为参数节点， Conv2d作为运算节点
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x_input_data')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_label_data')
# prediction result
y_true_cls = tf.argmax(y_true, axis=1)
# dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)

print("x:", x)
# one-hot code output
print("y_true:", y_true)
print("y_true_cls:", y_true_cls)
print("dropout probability:", keep_prob)

# softmax layer
# mnist data input is a 1D vector with length 784(28 * 28 pixels)
# reshape to match picture format [height * width * channel]
# tensor input become 4D: [batch_size, height, width, channel]
# -1 means: num_images/batch_size will be calculate automatically    
# reshpe the data
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# collect some input images which will be shown on tensorboard
tf.summary.image('input_images', x_image, 9)

# create the network
output_layer = conv_net(input=x_image,
                        weights=weights_dict,
                        biases=biases_dict,
                        dropout_prob=dropout_prob_1)

print("output_layer:", output_layer)

with tf.name_scope("softmax"):
    # CNN_model
    y_pred = tf.nn.softmax(output_layer)
print("y_pred:", y_pred)

y_pred_cls = tf.argmax(y_pred, axis=1)
print("y_pred_cls:", y_pred_cls)

########################################################################
# Loss 
########################################################################
# Compute cross entropy as our loss function
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,
#                                                           labels=y_true)
# print("cross_entropy:", cross_entropy)
# cross_entropy: Tensor("softmax_cross_entropy_with_logits/Reshape_2:0", shape=(?,), dtype=float32)

# 模型参数，包括 weight 和 bias
# trainable_vars = tf.trainable_variables()

# add name scope for tensorboard
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y_true))
    # L2正则化效果不好，accuracy~=97.0%
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y_true))+\
    #    BETA * tf.add_n([tf.nn.l2_loss(v)
    #                     for v in trainable_vars if not 'b' in v.name])

    # create a summary to monitor cost tensor on tensorboard
    tf.summary.scalar('cross_entropy', cost)

print("cost:", cost)
########################################################################
# accuracy
########################################################################
# add name scope for tensorboard
# use an AdamOptimizer to train the network
with tf.name_scope("optimizer_adam"):
    # optimization algorithm
    optimizer_adam = tf.train.AdamOptimizer(learning_rate=learning_rate_1).minimize(cost)
    # Gradiebt Descent
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate_1)
    # Op to calculate every variable gradient
    # grads = tf.gradients(cost, tf.trainable_variable())
    # grads = list(zip(grads, tf.trainable_variable()))
    # Op to update all variables according to their gradients
    # apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

# optimizer_gradientDescent = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_1).minimize(cost)

# now we use AdamOptimizer
optimizer = optimizer_adam
print("optimizer:", optimizer)
########################################################################
# optimization algorithme
########################################################################
# add name scope for tensorboard
# compute the accuracy
with tf.name_scope("accuracy"):
    # 1D tensor of type bool 
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    # cast the "correct_prediction" to type float32, then calculate the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # createb a summary to monitor accuracy tensor on tensorboard
    tf.summary.scalar('accuracy', accuracy)

print("correct_prediction:", correct_prediction)
print("accuracy:", accuracy)
########################################################################
# tensorflow session configration
########################################################################

import os

# "0,1,2" -> use the '/gpu:0' '/gpu:1' '/gpu:2'.
# "" -> no GPU visiable 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
# program can only occupy up to 90% gpu memory of '/gpu:0'
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# program apply memory on demande
config.gpu_options.allow_growth = True
# to find out which devices your operations and tensors are assigned to,
# create a session with 'log_device_placement' configuration option set to True.
config.log_device_placement = True
session = tf.Session(config=config)


# initialize all the virables
def init_variables():
    session.run(tf.global_variables_initializer())


# execute the function to initialize the variables
init_variables()

# create summaries to visualize weights
# for var in tf.trainable_variables():
#    tf.summary.histogram(var.name, var)

#  summarize all gradients
# for grad, var in grads:
#    tf.summary.histogram(var.name+"/gradient", grad)
########################################################################
# merge all the logs
########################################################################
# LOGDIR = BASEDIR + 'lr={:.0E},bs={}'.format(learning_rate, batch_size)

# tensorboard collect the summaries
logs_path = "logs/simple_cnn/2019-03-27"
# merge all summary into a single op
merged_summary = tf.summary.merge_all()
# write logs to tensorboard
writer = tf.summary.FileWriter(logs_path, session.graph)
# writer.add_graph(tf_session.graph)

########################################################################
# test some helper function
########################################################################
# from modules.helper_functions import predict_cls_test
# from modules.helper_functions import predict_cls_validation

# predict_cls_test(tf_session=session,
#                 x_ph=x,
#                 y_true_ph=y_true,
#                 keep_prob_ph=keep_prob,
#                 y_pred_cls_tensor = y_pred_cls)

# predict_cls_validation(tf_session=session,
#                       x_ph=x,
#                       y_true_ph=y_true,
#                       keep_prob_ph=keep_prob,
#                       y_pred_cls_tensor=y_pred_cls)

# test plot images function
# from modules.helper_functions import plot_images

# test the plot_images() funnction
# images = data.x_test[0:9]
# cls_true = data.y_test_cls[0:9]
# plot_images(images=images, cls_true=cls_true)
########################################################################
# weights and biases saver
########################################################################
# function for saving files during optimization iterations
# overwrite the saving file when we get a better accuracy
import tensorflow as tf
import os

save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'model.ckpt')
print("model_path:", model_path)

saver = tf.train.Saver(max_to_keep=5)
########################################################################
# performe optimization
########################################################################

#### accuracy before any optimization
# %matplotlib inline

from modules.predict_functions import print_test_accuracy

# calculate the predicted class for the test-set
print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_confusion_matrix=True)

#### performace after 1 optimization  iteration
from modules.predict_functions import optimize

# def main():
#    for lr in [1e-2, 1e-3, 1e-4]:
#        for bs in [64, 128]:
#            logging.info('learing rate = {:.0E}, batch size = {}'.format(lr, bs))
# cifar10_model(lr, bs)

# get the save path returned by saver.save() in optimization() methode
save_path = optimize(tf_session=session,
                     tf_saver=saver,
                     tensorboard_summary=merged_summary,
                     tensorboard_writer=writer,
                     model_path=model_path,
                     x_ph=x,
                     y_true_ph=y_true,
                     keep_prob_ph=keep_prob,
                     y_pred_cls_tensor=y_pred_cls,
                     optimizer=optimizer,
                     display_step=display_step,
                     num_iterations=1,
                     loss_opt=cost,
                     accuracy=accuracy,
                     dropout_prob=dropout_prob_1)
from modules.predict_functions import print_test_accuracy

print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_example_error=True,
                    show_confusion_matrix=True)
#### performace after 100 optimization  iteration
optimize(tf_session=session,
         tf_saver=saver,
         tensorboard_summary=merged_summary,
         tensorboard_writer=writer,
         model_path=model_path,
         x_ph=x,
         y_true_ph=y_true,
         keep_prob_ph=keep_prob,
         y_pred_cls_tensor=y_pred_cls,
         optimizer=optimizer,
         display_step=display_step,
         num_iterations=99,
         loss_opt=cost,
         accuracy=accuracy,
         dropout_prob=dropout_prob_1)

print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_confusion_matrix=True)

#### performace after 1000 optimization  iteration
optimize(tf_session=session,
         tf_saver=saver,
         tensorboard_summary=merged_summary,
         tensorboard_writer=writer,
         model_path=model_path,
         x_ph=x,
         y_true_ph=y_true,
         keep_prob_ph=keep_prob,
         y_pred_cls_tensor=y_pred_cls,
         optimizer=optimizer,
         display_step=display_step,
         num_iterations=900,
         loss_opt=cost,
         accuracy=accuracy,
         dropout_prob=dropout_prob_1)

print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_confusion_matrix=True)
#### performace after 10000 optimization  iteration
optimize(tf_session=session,
         tf_saver=saver,
         tensorboard_summary=merged_summary,
         tensorboard_writer=writer,
         model_path=model_path,
         x_ph=x,
         y_true_ph=y_true,
         keep_prob_ph=keep_prob,
         y_pred_cls_tensor=y_pred_cls,
         optimizer=optimizer,
         display_step=display_step,
         num_iterations=9000,
         loss_opt=cost,
         accuracy=accuracy,
         dropout_prob=dropout_prob_1)
print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_confusion_matrix=True)
########################################################################
# use saver to reload model
########################################################################
#### Saver
# reinitialize all the Variables of NN with random values
init_variables()
# print test accuracy
print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_confusion_matrix=True)
# reload all the variables that were saved to file during optimization
# get the latest model which has been saved
model_file = tf.train.latest_checkpoint('checkpoints')
saver.restore(sess=session, save_path=model_file)
print("Model restored from file: %s" % model_file) \
    # print test accuracy
print_test_accuracy(tf_session=session,
                    x_ph=x,
                    y_true_ph=y_true,
                    keep_prob_ph=keep_prob,
                    y_pred_cls_tensor=y_pred_cls,
                    show_confusion_matrix=True)
# close the session
session.close()
########################################################################
# visualizing the TF graph
########################################################################
# writer = tf.summary.FileWriter("tmp/simple_cnn/3")
# writer.add_graph(session.graph)

# in you commande line console, just write it down
# tensorboard --logdir ./tmp/simple_cnn/2
