#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import shap
import json
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

image_name = 'soccer'
X0 = image.load_img(os.path.join('data',image_name+'.jpg'), target_size=(224, 224))
plt.imshow(X0)
X0 = image.img_to_array(X0)
X0 = np.expand_dims(X0,axis=0)
model = VGG19(weights='imagenet',include_top=True)
X,y = shap.datasets.imagenet50()
to_explain = X0
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)
# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

e = shap.GradientExplainer(
    (model.layers[0].input, model.layers[-1].output),
    map2layer(preprocess_input(X.copy()), 0),
    local_smoothing=0 # std dev of smoothing noise
)

shap_values,indexes = e.shap_values(map2layer(to_explain, 0), ranked_outputs=1)
# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)
import cv2
a  = shap_values[0][0]
b = np.mean(a,axis=2)
np.savetxt(image_name+"shap_explain.csv",b,delimiter=',')

