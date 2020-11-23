#####Sanity check######

#Load packages
import sys
sys.path.append('../')
from methods.LEGv0 import *
#FOLDER_PATH = 'C:/Users/luosh/Desktop/LEG/Image'
FOLDER_PATH = '/CCAS/home/shine_lsy/LEG/imagenet/realworldimg/'
img_names = []
path_list = list(Path(FOLDER_PATH).rglob('*'+'jpg'))
for i in range(len(path_list)):
    img_names.append(path_list[i].stem)
layer_names = ["predictions","fc2","fc1","block5_conv4","block5_conv3","block5_conv2","block5_conv1","block4_conv4","block4_conv3","block4_conv2","block4_conv1",
              "block3_conv4","block3_conv3","block3_conv2","block3_conv1","block2_conv2","block2_conv1","block1_conv2","block1_conv1 "]

VGG19_MODEL = VGG19(include_top =True)
my_weights = VGG19_MODEL.get_weights()
layer_len = len(my_weights)+1
IMG_NUM = 15
NUM_SAMPLE = 10000
save_path = "result/LEGimagenet/"
########Cascading Randomization##########
for index in range(IMG_NUM):
    new_weights = my_weights.copy()
    img = image.load_img(path_list[index], target_size=(224,224))
    img = image.img_to_array(img).astype(int)
    image_input = np.expand_dims(img.copy(), axis = 0)
    pred_paras= predict_vgg19(img,  VGG19_MODEL)
    
    for k in range(len(layer_names)+1):
        if k == 0:
            #task0 = LEG_explainer(image_input, VGG19_MODEL, predict_vgg19 ,noise_lvl = 0.02, num_sample = 2000, penalty = 'TV', lambda_arr= [0.3],pred_paras= pred_paras)
            task0 = LEG_explainer(image_input, VGG19_MODEL, predict_vgg19 ,noise_lvl = 0.02, num_sample = NUM_SAMPLE, penalty = None)
            #print("Original Explanation is :")
            fig, ax = plt.subplots()
            im = ax.imshow(task0[0][0],cmap='hot')
            fig.colorbar(im)
            fig.savefig(save_path+img_names[index]+"_imagenet_ori.jpg")
            np.savetxt(save_path+img_names[index]+"_imagenet_ori.txt", task0[0][0])
        else:
            layer_weight_size = my_weights[layer_len-2*k-1].shape
            layer_bias_size = my_weights[layer_len-2*k].shape
            new_layer_weight = np.random.uniform(low=-0.05, high=0.05, size=layer_weight_size)
            new_layer_bias = np.random.uniform(low=-0.05, high=0.05, size=layer_bias_size)
            new_weights[layer_len-2*k-1] = new_layer_weight
            new_weights[layer_len-2*k] = new_layer_bias
            ###########update the model###############
            VGG19_MODEL.set_weights(new_weights)  
            #task1 = LEG_explainer(image_input, VGG19_MODEL, predict_vgg19 ,noise_lvl = 0.02, num_sample = 2000, penalty = 'TV', lambda_arr= [0.3], pred_paras= pred_paras)
            task1 = LEG_explainer(image_input, VGG19_MODEL, predict_vgg19 ,noise_lvl = 0.02, num_sample = NUM_SAMPLE, penalty = None)
            #print('The ',k,'th layer explanation is:')
            fig, ax = plt.subplots()
            im = ax.imshow(task1[0][0],cmap='hot')
            fig.colorbar(im)
            fig.savefig(save_path+img_names[index]+"_imagenet_"+layer_names[k-1]+".jpg")
            np.savetxt(save_path+img_names[index]+"_imagenet_"+layer_names[k-1]+".txt", task1[0][0])
    VGG19_MODEL.set_weights(my_weights)  
