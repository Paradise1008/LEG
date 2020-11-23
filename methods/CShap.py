#C-Shapley implementation regression estimation with only squares structure

# Load packages

import os
import keras
from keras.applications.vgg19 import VGG19, decode_predictions, preprocess_input
from keras.preprocessing import image
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from skimage.io import imread
from scipy.linalg import null_space
from scipy.special import comb
from time import time
import cv2

#Customized functions

def my_conv(input_conv, size , channel):  ##Convolution input size from a*b*3 into a/s * b/s *3 and compress into one channel if channel=-1
    length,width,data_channel = input_conv.shape
    new_length = int(length/size)
    new_width = int(width/size)
    new_input  =  np.zeros((new_length,new_width))
    if channel == -1:
        for i in range(0,new_length):
            for j in range(0,new_width):
                new_input[i,j] = np.mean(input_conv[i*size:((i+1)*size),j*size:((j+1)*size),:])     
    else:
        for i in range(0,new_length):
            for j in range(0,new_width):
                new_input[i,j] = np.mean(input_conv[i*size:((i+1)*size),j*size:((j+1)*size),channel])     
    return(new_input)

def my_2ddeconv(input_conv, size):   #Deconvolution function 
    length,width = input_conv.shape
    new_length = int(length*size)
    new_width = int(width*size)
    new_input  =  np.zeros((new_length,new_width))
    for i in range(0,new_length):
        for j in range(0,new_width):
            new_input[i,j] = input_conv[i//size, j//size]       
    return(new_input)

def create_data_matrix(i,j,L,k):
    res = np.zeros((L,L))
    res += 1
    for row in range(max(i-k,0),min(i+k+1,L)):
        for col in range(max(j-k,0),min(j+k+1,L)):
            if abs(row-i)+abs(col-j) <= k:
                res[row,col] = 0
    return(res.flatten())



#create inputs
def Create_Inputs(image, model,chosen_class,conv, max_order, background=None):
    if background is None:
        background = image.mean((0,1))
    L = int(image.shape[0]/conv)
    D = L*L
    index  = np.zeros((L,L,max_order))
    X = np.zeros((max_order,D,D))
    F = np.zeros((max_order,D))
    for i in range(L):
        for j in range(L):
            for k in range(max_order):
                position = create_data_matrix(i,j,L,k)
                image_t = image.copy()
                image_t[my_2ddeconv(position.reshape((L,L)),conv) == 1] = background
                #plt.imshow(image_t/255.0)
                #plt.show()
                image_input = np.expand_dims(image_t.copy(),axis=0)
                image_input = preprocess_input(image_input)
                preds = model.predict(image_input)
                F[k,i*L+j] = preds[0,chosen_class]
                #print(F[k,i*L+j])
                X[k,i*L+j,:] = 1-position
                
    F_ans = F.flatten() 
    X_ans = X.reshape((max_order*D,D))
    return(F_ans, X_ans)

def Get_CShap(image0,vgg_model,chosen_class,conv=8,max_order=4):
    F,X = Create_Inputs(image0,vgg_model,chosen_class,conv=conv,max_order=max_order)
    L = int(image0.shape[0]/conv)
    D = L*L
    S = np.sum(X,axis=1)
    W = np.zeros(max_order*D)
    for i in range(max_order*D):
        W[i] = (D-1)/(comb(D,S[i])*S[i]*D)
    W = np.diag(W)
    score = inv(X.transpose()@W@X)@X.transpose()@F
    ht = cv2.resize(score.reshape((L,L)), (L*conv,L*conv))    
    #np.savetxt(image_name+ "C-Shap_explain.csv",ht,delimiter=',')
    return(ht)



# test a simple test image
if __name__ == "__main__":
    print("We are excuting CShap program", __name__)
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
    max_order =4
    conv =8 
    sample_size = np.square(224/8)*4
    CShap  = Get_CShap(img, VGG19_MODEL,  chosen_class)
    end_time = time()
    np.savetxt("Sample_Test_CShap.txt", CShap)
    plt.imshow(CShap, cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_CShap.png")
    print("CShap computed and saved successfully")
    print("The time spent on CShap explanation for approx.",sample_size, " is ",round((end_time - begin_time)/60,2), "mins") 

