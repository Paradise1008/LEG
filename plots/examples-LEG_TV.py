###Figure 2 Example Plots
import sys
import matplotlib.pyplot as plt
import matplotlib

sys.path.append('../')
from LEG.methods.LEGv0 import *
img_name = ["dogcat","icecream"]
# read the model
VGG19_MODEL = VGG19(include_top = True)
print("VGG19 has been imported successfully")

def plot_image(image):
    matplotlib.axes.Axes.clear()
    plt.imshow(image)
    plt.draw()
    plt.pause()
for i in range(len(img_name)):
    #img = image.load_img('/CCAS/home/shine_lsy/LEG/test/images/trafficlight.jpg', target_size=(224,224))
    img = image.load_img('/home/zhou/Documents/github/XAI/LEG/Image/'+img_name[i]+'.jpg', target_size=(224,224))
    img = image.img_to_array(img).astype(int)
    image_input = np.expand_dims(img.copy(), axis = 0)
    image_input = preprocess_input(image_input)
    print("Image has been read successfully")
    # make the prediction of the image by the vgg19
    preds = VGG19_MODEL.predict(image_input)
    for pred_class in decode_predictions(preds)[0]:
        print(pred_class)
    chosen_class = np.argmax(preds)
    print("The Classfication Category is ", chosen_class)
    begin_time = time()
    LEG_small = LEG_explainer(np.expand_dims(img.copy(), axis = 0), VGG19_MODEL,
                               predict_vgg19,base_size = 28, num_sample = 4000,
                               penalty=None, noise_lvl=0.02)
    LEGTV_small  = LEG_explainer(np.expand_dims(img.copy(), axis = 0), VGG19_MODEL,
                                 predict_vgg19,base_size = 28, num_sample = 4000,
                                 penalty='TV',lambda_arr = [0.3],noise_lvl=0.02)
    end_time = time()

    
    heatmap = LEG_small[0][0]/np.max(LEG_small[0][0])
    plt.imshow(img)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig("figure2_LEG_"+img_name[i]+".eps",bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()
    heatmap = LEGTV_small[0][0]/np.max(LEGTV_small[0][0])
    plt.imshow(img)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig("figure2_LEGTV_"+img_name[i]+".eps",bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()
    print("LEG computed and saved successfully")
    print("The time spent on LEG explanation is ",round((end_time - begin_time)/60,2), "mins")

