#GardCam implementation adopted by Keras-Gradcam 

#Load packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import cv2
from time import time
from keras.preprocessing import image
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as K
#Customized functions

def Get_GradCam(
    img_array, model, last_conv_layer_name = "conv2d_2", classifier_layer_names= ["max_pooling2d_2","flatten_1","dense_1","dropout_1","dense_2", "dropout_2", "dense_3", "activation_1"]):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (28,28)) #Can change the size to fit other models 
    return heatmap


# test a simple test image
if __name__ == "__main__":
    print("We are excuting GradCam program on MNIST dataset", __name__)
    # read the image and preprocess for all methods
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = (x_train - 0.5) * 2
    x_test = (x_test - 0.5) * 2
    num_classes = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    
    #Load the model and get a new one
    MNIST_MODEL = load_model("/CCAS/home/shine_lsy/LEG/mnist/model/MNIST_new_model_py.h5")
    begin_time = time()
    for i in range(100):
        GradCam  = Get_GradCam(x_test[i:i+1], model = MNIST_MODEL)
        np.savetxt("/CCAS/home/shine_lsy/LEG/mnist/result/GradCam/"+str(i)+'.txt', GradCam)
    end_time = time()  
    print("GradCam computed and saved successfully")
    print("The time spent on GradCam explanation is ",round((end_time - begin_time)/60,2), "mins") 
   
