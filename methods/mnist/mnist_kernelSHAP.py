# Implementation of SHAP method

# import packages

import keras
from keras.preprocessing import image
from keras.datasets import mnist
from keras import backend as K
import requests
from skimage.segmentation import slic
from keras.models import load_model
import numpy as np
import shap
from time import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pdf")
#Define the functions

# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation ==j, :] = background
    return out

# fill the segmentation by the values which indicate the number for every seg.
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


# present the KernelSHAP explainer
def Get_KernelSHAP(image0,model,nsamples, chosen_class, n_seg = 50): 
    def f(z):
         return model.predict(mask_image(z, segments_slic, img_orig, 255))
    img_orig = image0
    segments_slic = slic(image.array_to_img(image0), n_segments = n_seg, compactness = 30, sigma=3)
    explainer = shap.KernelExplainer(f, np.zeros((1,n_seg)))
    shap_values = explainer.shap_values(np.ones((1,n_seg)), nsamples = nsamples)
    m = fill_segmentation(shap_values[chosen_class][0], segments_slic)
    return(m)

# test a simple test image
if __name__ == "__main__":
    print("We are excuting KernelSHAP program on MNIST dataset", __name__)
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
        print(i)
        img = x_test[i:i+1]
        preds = MNIST_MODEL.predict(img)
        chosen_class = np.argmax(preds)
        kSHAP = Get_KernelSHAP(img[0], model = MNIST_MODEL, nsamples=60000,chosen_class = chosen_class )
        np.savetxt("/CCAS/home/shine_lsy/LEG/mnist/result/KSHAP50/"+str(i)+'.txt', kSHAP)
    end_time = time()  
    print("KernelShap computed and saved successfully")
    print("The time spent on GradCam explanation is ",round((end_time - begin_time)/60,2), "mins") 
