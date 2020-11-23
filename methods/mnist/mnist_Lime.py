#LIME implementation 

# Load packages
import os 
import keras
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image
from skimage.io import imread
from skimage.segmentation import mark_boundaries
import matplotlib
#matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lime 
from lime import lime_image
from time import time
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as K
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images
#LIME customized functions

from sklearn.pipeline import Pipeline
# stack many images
class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)

makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])


# Get the LIME explanations
def Get_Lime(image_input, mnist_model, nsamples, chosen_class, nfea = 10):

    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=0.001, max_dist=0.001, ratio=0.2)
    #explanation = explainer.explain_instance(image[0].astype('double'), mnist_model.predict, top_labels = 10, hide_color = 0, num_samples = nsamples)
    
    explanation = explainer.explain_instance(image_input[0], classifier_fn = mnist_model.predict, top_labels=10, hide_color=0, num_samples=nsamples, segmentation_fn=segmenter)

    temp, mask = explanation.get_image_and_mask(chosen_class, positive_only=False, num_features=nfea, hide_rest=False, min_weight = 0.01)
    return(temp, mask)

# test a simple test image
if __name__ == "__main__":
    print("We are excuting LIME program on MNIST dataset", __name__)
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
    dense_input = keras.layers.Input(shape=(28, 28, 3))
    dense_filter = keras.layers.Conv2D(1, 1, padding='same')(dense_input)
    output = MNIST_MODEL(dense_filter)
    new_mnist= keras.Model(dense_input, output)
    new_mnist.compile(loss='mse', optimizer='adam')
    weights = new_mnist.get_weights()
    weights[0] = [[[[0.2125],[0.7154],[0.0721]]]]
    weights[1] = [0.]
    new_mnist.set_weights(weights)  
    
    #Get the prediction and save the explanations
    begin_time = time()
    for i in range(100):
        print(i)
        new_image_input = gray2rgb(x_test[i:i+1].reshape((-1,28,28)))
        preds = new_mnist.predict(new_image_input )
        chosen_class = np.argmax(preds)
        
        sample_size = 20000
        temp, mask = Get_Lime(new_image_input, new_mnist, sample_size , chosen_class)
        plt.imshow(mask)
        plt.show()
        np.savetxt("/CCAS/home/shine_lsy/LEG/mnist/result/LIME_MORE/"+str(i)+'.txt', mask)
    end_time = time()
  
