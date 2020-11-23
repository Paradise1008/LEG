#LIME implementation 

# Load packages
import os 
import keras
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image
from skimage.io import imread
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lime 
from lime import lime_image
from time import time


#LIME customized functions

# stack many images
def transform_img_fn(image_input):
    out = []
    out.append(image_input)
    return np.vstack(out)

# Get the LIME explanations
def Get_Lime(image_input, vgg_model, nsamples, chosen_class, nfea = 5):
    images = transform_img_fn(image_input)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype('double'), vgg_model.predict, top_labels = 1, hide_color = 0, num_samples = nsamples)
    temp, mask = explanation.get_image_and_mask(chosen_class, positive_only= False, num_features = nfea, hide_rest = False)
    return(mask)



# test a simple test image
if __name__ == "__main__":
    print("We are excuting LIME program", __name__)
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
    LIME  = Get_Lime(image_input, VGG19_MODEL, sample_size , chosen_class)
    end_time = time()
    np.savetxt("Sample_Test_LIME.txt", LIME)
    plt.imshow(LIME, cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_LIME.png")
    print("KernelSHAP computed and saved successfully")
    print("The time spent on LIME explanation for ",sample_size, " is ",round((end_time - begin_time)/60,2), "mins") 
