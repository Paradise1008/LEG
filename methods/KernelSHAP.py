# Implementation of SHAP method

# import packages

import keras
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image

import requests
from skimage.segmentation import slic
import numpy as np
import shap
from time import time
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

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
         return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))
    img_orig = image0
    segments_slic = slic(image.array_to_img(image0), n_segments = n_seg, compactness = 30, sigma=3)
    explainer = shap.KernelExplainer(f, np.zeros((1,n_seg)))
    shap_values = explainer.shap_values(np.ones((1,n_seg)), nsamples = nsamples)
    m = fill_segmentation(shap_values[chosen_class][0], segments_slic)
    return(m)

# test a simple test image
if __name__ == "__main__":
    print("We are excuting program", __name__)
    # read the image
    img = image.load_img('/CCAS/home/shine_lsy/LEG/test/images/trafficlight.jpg', target_size=(224,224))
    img = image.img_to_array(img).astype(int)
    image_input = np.expand_dims(img.copy(), axis = 0)
    image_input = preprocess_input(image_input)
    print("Image has been read successfully")

    # read the model
    VGG19_MODEL = VGG19(include_top = True)
    print("VGG19 has been imported successfully")
    # make the prediction of the image by the vgg19
    preds = VGG19_MODEL.predict(image_input)
    for pred_class in decode_predictions(preds)[0]:
        print(pred_class)
    chosen_class = np.argmax(preds)
    print("The Classfication Category is ", chosen_class)
    begin_time = time()
    sample_size = 3000
    KernelSHAP = Get_KernelSHAP(img, VGG19_MODEL, sample_size , chosen_class)
    end_time = time()
    np.savetxt("Sample_Test_KernelSHAP.txt", KernelSHAP)
    plt.imshow(KernelSHAP, cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_KernelSHAP.png")
    print("KernelSHAP computed and saved successfully")
    print("The time spent on KernelSHAP explanation for ",sample_size, " is ",round((end_time - begin_time)/60,2), "mins") 
