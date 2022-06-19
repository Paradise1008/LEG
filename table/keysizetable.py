###ImageNet Table Statistics#####
import sys
import time

sys.path.append('../')
from LEG.methods.LEGv0 import *

if __name__ == "__main__":
    FOLDER_PATH = '/CCAS/home/shine_lsy/LEG/imagenet/'
    IMG_NUM = 500
    VGG19_MODEL = vgg19.VGG19(include_top=True)
    print("Model Imported")
    ######################################################################
    METHODS = {'LEG': 'LEG', 'LEG-TV': 'LEGv0', 'LIME': 'Lime', 'CShap': 'CShap', 'KernelSHAP': 'KernelSHAP',
               'GradCam': 'GradCam'}
    backgrounds = [-100, -1000, None, 255, 0]
    names = ["dist-100", "noise", "Mean", "white-out", "black-out"]
    alpha_c = np.linspace(0, 1, 1000)
    for background, name in zip(backgrounds, names):
        df = pd.DataFrame({'x': np.arange(IMG_NUM)})
        for key in METHODS:
            a = np.zeros(IMG_NUM) + 1  # set a to 1 at first
            for k in range(IMG_NUM):
                heatmap = np.loadtxt(FOLDER_PATH + 'result/' + METHODS[key] + '/' + str(k) + '.txt')
                ori_img = image.load_img(FOLDER_PATH + 'images/' + str(k) + '.png', target_size=(224, 224))
                ori_img = image.img_to_array(ori_img).astype(int)
                if background == None:
                    background2 = ori_img.mean((0, 1))
                else:
                    background2 = background
                ori_category = predict_vgg19(ori_img, VGG19_MODEL).category
                for alpha in alpha_c:
                    perturb_img = get_mask(ori_img, heatmap, alpha, background=background2, sort_mode='abs')
                    new_category = predict_vgg19(perturb_img, VGG19_MODEL).category
                    if new_category == ori_category:
                        continue
                    else:
                        a[k] = alpha
                        break
            thisdict = dict(key=a, x=np.arange(IMG_NUM))
            thisdict[key] = thisdict.pop('key')
            df_temp = pd.DataFrame(thisdict)
            df = pd.merge(df, df_temp, on='x')
        df.to_csv("new_keysize_imagenet" + name + ".csv")
        df.describe().to_csv("new_keysize_stats_imagenet" + name + ".csv")
