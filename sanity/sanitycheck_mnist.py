#Sanity Check in MNIST dataset##
import sys
sys.path.append('../')
from methods.LEGv0 import *
from keras.models import load_model
from keras.datasets import mnist
import keras
from keras import backend as K

#load_model and get test_data
MNIST_MODEL = load_model("/CCAS/home/shine_lsy/LEG/mnist/model/MNIST_new_model_py.h5")
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
num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
IMG_NUM = 5
INPUTS = x_test[:IMG_NUM]
INPUTS2 = INPUTS.copy()
INPUTS2 = ((INPUTS2-127.5)*0.5+127.5).astype("int")


layer_names = ["dense_3","dense_2","dense_1","conv2d_2","conv2d_1"]
save_path = 'result/mnist/'
my_weights = MNIST_MODEL.get_weights()
layer_len = len(my_weights)+1
IMG_NUM = 10
NUM_SAMPLE =100000
########Cascading Randomization##########
for index in range(0,IMG_NUM):
    new_weights = my_weights.copy()
    image_input = INPUTS2[index:index+1]
    pred_paras= predict_MNIST(image_input[0],  MNIST_MODEL)

    for k in range(len(layer_names)+1):
        if k == 0:
            #task0 = LEG_explainer(image_input, MNIST_MODEL, predict_MNIST ,noise_lvl = 0.02, num_sample = 2000, penalty = 'TV', lambda_arr= [0.3],pred_paras= pred_paras)
            task0 = LEG_explainer(image_input, MNIST_MODEL, predict_MNIST ,noise_lvl = 0.02, num_sample = NUM_SAMPLE, penalty =None)
            #print("Original Explanation is :")
            fig, ax = plt.subplots()
            im = ax.imshow(task0[0][0],cmap='hot')
            fig.colorbar(im)
            fig.savefig(save_path+str(index)+"_mnist_ori.jpg")
            np.savetxt(save_path+str(index)+"_mnist_ori.txt", task0[0][0])
        else:
            layer_weight_size = my_weights[layer_len-2*k-1].shape
            layer_bias_size = my_weights[layer_len-2*k].shape
            new_layer_weight = np.random.uniform(low=-0.05, high=0.05, size=layer_weight_size)
            new_layer_bias = np.random.uniform(low=-0.05, high=0.05, size=layer_bias_size)
            new_weights[layer_len-2*k-1] = new_layer_weight
            new_weights[layer_len-2*k] = new_layer_bias
            ###########update the model###############
            MNIST_MODEL.set_weights(new_weights)  
            #task1 = LEG_explainer(image_input, MNIST_MODEL, predict_MNIST ,noise_lvl = 0.02, num_sample = 2000, penalty = 'TV', lambda_arr= [0.3], pred_paras= pred_paras)
            task1 = LEG_explainer(image_input, MNIST_MODEL, predict_MNIST ,noise_lvl = 0.02, num_sample = NUM_SAMPLE, penalty = None)
            #print('The ',k,'th layer explanation is:')
            fig, ax = plt.subplots()
            im = ax.imshow(task1[0][0],cmap='hot')
            fig.colorbar(im)
            fig.savefig(save_path+str(index)+"_mnist_"+layer_names[k-1]+".jpg")
            np.savetxt(save_path+str(index)+"_mnist_"+layer_names[k-1]+".txt", task1[0][0])
    MNIST_MODEL.set_weights(my_weights)  
