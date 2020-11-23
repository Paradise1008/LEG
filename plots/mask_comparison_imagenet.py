##Make the plots with masked parts with different schemes
import sys
sys.path.append('../')
from methods.LEGv0 import *
from matplotlib import gridspec
##########

VGG19_MODEL = VGG19(include_top = True)
#picture_id =  np.arange(10)+100
picture_id =  [7, 11, 24,41,51,79,89,93,98,108]
method_id = ["Origin","LEG","LEGv0","KernelSHAP","Lime","CShap","GradCam"]
cols = ["Origin","LEG","LEG-TV","KernelSHAP","LIME","C-Shap","GradCam"]
rows = []
PLOT_DATA = []

for i in range(len(picture_id)):
    img= image.load_img('C:/Users/luosh/Desktop/CVPR-LEG-CODE/imagenet/images/'+str(picture_id[i])+'.png', target_size=(224,224))
    img = image.img_to_array(img).astype(int)
    image_input = np.expand_dims(img.copy(), axis = 0)
    image_input = preprocess_input(image_input)
    preds = VGG19_MODEL.predict(image_input)
    for pred_class in decode_predictions(preds)[0]:
        print(pred_class)
    rows.append(str(i)+decode_predictions(preds)[0][0][1])
    chosen_class = np.argmax(preds)
    print("The Classfication Category is ", chosen_class)
    for j in range(len(method_id)):
        if j==0:
            PLOT_DATA.append(img)
        else:
            heatmap = np.loadtxt('C:/Users/luosh/Desktop/CVPR-LEG-CODE/imagenet/result/'+method_id[j]+"/"+str(picture_id[i])+'.txt')
            mask_img = get_mask(img, heatmap, alpha = 0.1, background = 0, sort_mode = 'abs')
            PLOT_DATA.append(mask_img)
ncol = 7
nrow = len(picture_id)
"""
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, constrained_layout=False,subplot_kw = {'xticks':[], 'yticks':[]},figsize=(2*ncol, 2*nrow))
for i in range(nrow):
    for j in range(ncol):
        if j == 0: 
            hp = axes[i,j].imshow(PLOT_DATA[i*(ncol)+j])
        else:
            hp = axes[i,j].imshow(PLOT_DATA[i*(ncol)+j],cmap='hot', interpolation='nearest')
for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90)

fig.tight_layout()
#fig.savefig("sample.png")
plt.show()
"""
######################################
count = 0
fig = plt.figure(figsize=(18,28))
gs = gridspec.GridSpec(nrow, ncol,
                       wspace=0.0, hspace=0.0)
for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[i, j])
        if (count==0) or (count%ncol==0):
            ax.imshow(PLOT_DATA[count])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.imshow(PLOT_DATA[count])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        # add labels
        if count < ncol:
            ax.set_title(cols[count], fontsize=28)
        count +=1
        if ax.is_first_col():
            ax.set_ylabel(rows[i], fontsize=18)
plt.savefig("sanity_compacttry.png")
plt.show()



"""
##########################################################################################
##########################################################################################
picture_id =  [346,347]
method_id = [None,-50,-100,None,-256,255,0]
cols = ["Origin","Dist-50","Dist-100","Mean","Noise","White","Black"]
rows = []
PLOT_DATA = []
for i in range(len(picture_id)):
    img= image.load_img('C:/Users/luosh/Desktop/CVPR-LEG-CODE/imagenet/images/'+str(picture_id[i])+'.png', target_size=(224,224))
    img = image.img_to_array(img).astype(int)
    image_input = np.expand_dims(img.copy(), axis = 0)
    image_input = preprocess_input(image_input)
    preds = VGG19_MODEL.predict(image_input)
    for pred_class in decode_predictions(preds)[0]:
        print(pred_class)
    rows.append(decode_predictions(preds)[0][0][1])
    chosen_class = np.argmax(preds)
    print("The Classfication Category is ", chosen_class)
    for j in range(len(method_id)):
        if j==0:
            PLOT_DATA.append(img)
        else:
            if method_id[j] == None:
                background = img.mean((0, 1))
            else:
                background = method_id[j]
            heatmap = np.loadtxt('C:/Users/luosh/Desktop/CVPR-LEG-CODE/imagenet/result/LEGv0/'+str(picture_id[i])+'.txt')
            mask_img = get_mask(img, heatmap, alpha = 0.1, background = background, sort_mode = 'abs')
            PLOT_DATA.append(mask_img)
ncol = len(cols)
nrow = len(picture_id)

#plt.rc('axes', titlesize=24)     # fontsize of the axes title
#plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, constrained_layout=False,subplot_kw = {'xticks':[], 'yticks':[]},figsize=(2*ncol, 2*nrow))
for i in range(nrow):
    for j in range(ncol):
        if j == 0: 
            hp = axes[i,j].imshow(PLOT_DATA[i*(ncol)+j])
        else:
            hp = axes[i,j].imshow(PLOT_DATA[i*(ncol)+j],cmap='hot', interpolation='nearest')
for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90)

fig.tight_layout()
fig.savefig("sample2.png")
plt.show()


count = 0
fig = plt.figure(figsize=(18,5.4))
gs = gridspec.GridSpec(nrow, ncol,
                       wspace=0.0, hspace=0.0)
for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[i, j])
        if (count==0) or (count%ncol==0):
            ax.imshow(PLOT_DATA[count])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.imshow(PLOT_DATA[count])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        # add labels
        if count < ncol:
            ax.set_title(cols[count], fontsize=28)
        count +=1
        if ax.is_first_col():
            ax.set_ylabel(rows[i], fontsize=18)
plt.savefig("sample_compact.png")
plt.show()
"""