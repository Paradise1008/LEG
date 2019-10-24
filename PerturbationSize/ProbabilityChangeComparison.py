#import related package
import keras
from keras.applications import vgg19 
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np
import cv2
import matplotlib.pyplot as plt

##########Display the original image##########
filename = "soccer"
path =os.path.join("data",filename+".jpg")  

from keras.preprocessing import image
X0 = image.load_img(path, target_size=(224, 224))
X0 = image.img_to_array(X0)

r1 = np.genfromtxt(filename+"0.0001LEG.csv",dtype=None, delimiter = ",") #our_explain.csv
r2 = np.genfromtxt(filename+"0.075LEG.csv",dtype=None, delimiter = ",") #our_explain.csv
lime = np.genfromtxt(filename+"lime_explain.csv",dtype=None,delimiter = ",")
shap = np.genfromtxt(filename+"shap_explain.csv",dtype=None,delimiter = ",")
heatmap_g = np.genfromtxt(filename+"grad_explain.csv",dtype=None,delimiter = ",")

def get_mask(X1,heatmap,alpha=0.1):
    result = X1.copy()
    temp = np.zeros(224*224)+255
    heat_flat = heatmap.reshape(1,224*224)
    order = np.argsort(-heat_flat)
    for m in range(0,round(alpha*224*224)):
        ind = order[0,m]
        temp[ind] = 0
    temp = temp.reshape(224,224)
    for i in range(0,224):
        for j in range(0,224):
            for k in range(0,3):
                result[i,j,k] = np.minimum(temp[i,j],result[i,j,k])
    return  result

vgg_model = vgg19.VGG19(include_top =True)
ori_img = vgg19.preprocess_input(np.expand_dims(X0.copy(),axis=0))
preds0 = vgg_model.predict(ori_img)

alpha_level = 0.1
chosen_class = np.argmax(preds0)

new_lime = get_mask(X0,lime,alpha=alpha_level)
plt.imshow(new_lime/255.0)
plt.imsave('Result/'+filename+"_lime_"+str(alpha_level)+"mask.jpg", new_lime/255.0)

new_grad = get_mask(X0,heatmap_g,alpha=alpha_level)
plt.imshow(new_grad/255.0)
plt.imsave('Result/'+filename+"_gradcam_"+str(alpha_level)+"mask.jpg", new_grad/255.0)

new_shap = get_mask(X0,shap,alpha=alpha_level)
plt.imshow(new_shap/255.0)
plt.imsave('Result/'+filename+"_shap_"+str(alpha_level)+"mask.jpg", new_shap/255.0)

new_leg1 = get_mask(X0,r1,alpha=alpha_level)
plt.imshow(new_leg1/255.0)
plt.imsave('Result/'+filename+"_LEGAIC_"+str(alpha_level)+"mask.jpg", new_leg1/255.0)

new_leg2 = get_mask(X0,r2,alpha=alpha_level)
plt.imshow(new_leg2/255.0)
plt.imsave('Result/'+filename+"_LEGCustomAIC_"+str(alpha_level)+"mask.jpg", new_leg2/255.0)

import pandas as pd

####################Mask Procedure#####################
alpha_c = np.arange(0.01,0.8,0.01)
num = alpha_c.shape[0]
lime_data = np.zeros(num)
shap_data = np.zeros(num)
grad_data = np.zeros(num)
LEG1_data = np.zeros(num)
LEG2_data = np.zeros(num)


for i in range(num):
    lime_data[i] = vgg_model.predict(vgg19.preprocess_input(np.expand_dims(get_mask(X0.copy(),lime,alpha=float(alpha_c[i])),axis=0)))[0,chosen_class]
    shap_data[i] = vgg_model.predict(vgg19.preprocess_input(np.expand_dims(get_mask(X0.copy(),shap,alpha=float(alpha_c[i])),axis=0)))[0,chosen_class]
    grad_data[i] = vgg_model.predict(vgg19.preprocess_input(np.expand_dims(get_mask(X0.copy(),heatmap_g,alpha=float(alpha_c[i])),axis=0)))[0,chosen_class]
    LEG1_data[i] = vgg_model.predict(vgg19.preprocess_input(np.expand_dims(get_mask(X0.copy(),r1,alpha=float(alpha_c[i])),axis=0)))[0,chosen_class]
    LEG2_data[i] = vgg_model.predict(vgg19.preprocess_input(np.expand_dims(get_mask(X0.copy(),r2,alpha=float(alpha_c[i])),axis=0)))[0,chosen_class]
# Data
import pandas as pd
df=pd.DataFrame({'percentage': alpha_c, 'lime': lime_data,  'shap': shap_data , 'grad': grad_data , 'LEG_exact': LEG1_data, 'LEG': LEG2_data})
df.to_csv(filename+"resultcompare.csv")
# multiple line plot
plt.plot( 'percentage', 'lime', data=df, marker='', color='green', linewidth=2, linestyle='solid', label="lime")
plt.plot( 'percentage', 'shap', data=df, marker='', color='red', linewidth=2, linestyle='solid', label="shap")
plt.plot( 'percentage', 'grad', data=df, marker='', color='yellow', linewidth=2, linestyle='solid', label="grad")
plt.plot( 'percentage', 'LEG_exact', data=df, marker='', color='purple', linewidth=2, linestyle='solid', label="LEG_exact")
plt.plot( 'percentage', 'LEG', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label="LEG")
plt.legend()
plt.savefig('Result/'+filename+"Compare.jpg")
plt.show()    
    
    
    
    



