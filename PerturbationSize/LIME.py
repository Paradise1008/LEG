#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import keras
from keras.applications import vgg19
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
vgg_model = vgg19.VGG19(include_top =True)

image_name = "soccer"


# In[ ]:


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = vgg19.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

images = transform_img_fn([os.path.join('data',image_name+'.jpg')])
#plt.imshow(image.load_img(img_path, target_size=(224,224)))
preds = vgg_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)
    
chosen_class = np.argmax(preds)
chosen_class
preds[0,chosen_class]

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..','..'))
    import lime
from lime import lime_image



explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(images[0],vgg_model.predict, top_labels =1 , hide_color = 0, num_samples =500)

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(chosen_class,positive_only=False, num_features = 1, hide_rest = False)
np.savetxt(image_name+ "lime_explain.csv",mask,delimiter=',')


