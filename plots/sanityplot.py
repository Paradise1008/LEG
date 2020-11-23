from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
####Cascading randomization plot####
col_names = ["inputs","ori","predictions","fc2","fc1","block5_conv4","block5_conv3","block5_conv2","block5_conv1","block4_conv4","block4_conv3","block4_conv2","block4_conv1",
              "block3_conv4","block3_conv3","block3_conv2","block3_conv1","block2_conv2","block2_conv1","block1_conv2","block1_conv1 "]

col_names2 = ["inputs","ori","predictions","fc2","fc1","block5\nconv4","block5\nconv3","block5\nconv2","block5\nconv1","block4\nconv4","block4\nconv3","block4\nconv2","block4\nconv1",
              "block3\nconv4","block3\nconv3","block3\nconv2","block3\nconv1","block2\nconv2","block2\nconv1","block1\nconv2","block1\nconv1 "]

#col_names = ["inputs","ori","fc1","block5_conv1","block4_conv1","block3_conv1","block2_conv1","block1_conv1 "]
#col_names2 = ["inputs","ori","fc1","block5\nconv1","block4\nconv1","block3\nconv1","block2\nconv1","block1\nconv1 "]
#


row_names = ["Tortoise","trafficlight","pineapple","mailbox"]
nrows = len(row_names)
ncols = len(col_names)
fig = plt.figure(figsize=(40,8))
gs = gridspec.GridSpec(nrows, ncols,
                       wspace=0.0, hspace=0.0)
cmap='hot'
#cmap = 'rainbow'
count = 0
folder_path = 'C:/Users/luosh/Desktop/CVPR-LEG-CODE/sanity/result/LEGimagenet/'
master_mask_list_abs_norm = []
for i in range(nrows):
    for j in range(ncols):
        if col_names[j] == "inputs":
            img = image.load_img("C:/Users/luosh/Desktop/LEG/Image/"+row_names[i]+".jpg", target_size=(224,224))
            img = image.img_to_array(img).astype('uint8')
            master_mask_list_abs_norm.append(img)
        else:
            img= np.loadtxt(folder_path+row_names[i]+"_imagenet_"+col_names[j]+'.txt')
            master_mask_list_abs_norm.append(img)


for i in range(nrows):
    for j in range(ncols):
        ax = plt.subplot(gs[i, j])
        if (count==0) or (count%ncols==0):
            ax.imshow(master_mask_list_abs_norm[count])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            if (count%ncols==1):
                vmax = np.max(master_mask_list_abs_norm[count])
            vmax0 = np.max(master_mask_list_abs_norm[count])
            if vmax0 > vmax: 
                vmax_final = vmax0 
            else:
                vmax_final = vmax
            ax.imshow(master_mask_list_abs_norm[count],
                      vmin = 0,
                      vmax = vmax_final,
                      cmap=cmap)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        # add labels
        if count < ncols:
            ax.set_title(col_names2[count], fontsize=24)
        count +=1
plt.savefig("sanity_try2.png")
plt.show()