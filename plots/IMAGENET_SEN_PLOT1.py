###Make the MNIST dataset sensitivity ###
# Load packages
import sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
sys.path.append('../')
from methods.LEGv0 import *
import keras

# Customized functions

def get_odds(x):
    return(x/(1-x))

def get_log(x):
    return(np.log(x))

def create_LOR(df):
    nrow, ncol = df.shape
    df2 = df.copy()
    df2.iloc[:,1:ncol] = df2.iloc[:,1:ncol].apply(get_odds)
    for i in range(nrow):
        df2.iloc[nrow-i-1,1:]= df2.iloc[nrow-i-1,1:].div(df2.iloc[0,1:])
    df2.iloc[:,1:ncol] = df2.iloc[:,1:ncol].apply(get_log)
    return df2

def createsenPLOT(df,filename,colors=None,markers=None,line_styles=None,legend=True):
    if colors is None:
        colors = ['tab:red','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    if markers is None:
        markers = ['D','*','o','v','^','s','*','+']
    if line_styles is None:
        line_styles = ['-','--','dotted','-.','solid',(0, (3, 1, 1, 1))]
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6,4),dpi=400)
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    col_names = df.columns.values
    x_axis = col_names[0]
    methods = col_names[1:]
    plt.ylim(-8.0,0.0)
    for k in range(len(methods)):
        plt.plot(df[x_axis], df[methods[k]], linestyle= line_styles[k], marker=markers[k], color=colors[k], linewidth=1.5,label=methods[k])
    if legend:
        plt.legend( title="METHODS",loc=1, fontsize='small',bbox_to_anchor=(1.35, 1),fancybox=True, shadow=True)
    plt.ylabel("Log Odds Ratio", fontsize=18)
    plt.xlabel("Perturbation Size", fontsize=18)
    plt.savefig(filename, bbox_inches="tight")
    return "Completed"



if __name__ == "__main__":
    VGG19_MODEL = VGG19(include_top = True)
    #folder_path = '/CCAS/home/shine_lsy/LEG/imagenet/'
    folder_path = 'C:/Users/luosh/Desktop/CVPR-LEG-CODE/imagenet/'
    print("Model imported")
    IMG_NUM = 5
    #IMG_NUM = 500
    markers = ['']*10
    METHODS = {'LEG':'LEG' , 'LEG-TV':'LEGv0', 'LIME':'Lime'  , 'CShap':'CShap' , 'KernelSHAP':'KernelSHAP', 'GradCam': 'GradCam'}
    INDEX = np.arange(IMG_NUM)
    alpha_c = np.linspace(0,0.4,10)
    #alpha_c = np.linspace(0,0.4,40)
    background = [-50,-255,None,-1000]
    for i in range(4):
        print("Sensitivity Analysis"+str(i)+'/4 begins')
        df=pd.DataFrame({'x' : alpha_c})
        a=np.zeros(len(alpha_c))
        for key in METHODS:
            a=np.zeros(len(alpha_c))
            for k in range(IMG_NUM):
                ori_img = image.load_img(folder_path+'images/'+str(k)+'.png', target_size=(224,224))
                ori_img = image.img_to_array(ori_img).astype(int)
                heatmap = np.loadtxt(folder_path+'result/'+METHODS[key]+'/'+str(k)+'.txt')
                if background[i] == -1000:
                    df_temp = sensitivity_anal(predict_vgg19 , ori_img, heatmap, VGG19_MODEL, alpha_c=alpha_c , sort_mode='abs', background= background[i], Show_Option = False, repeat=2)
                else:
                    df_temp = sensitivity_anal(predict_vgg19 , ori_img, heatmap, VGG19_MODEL, alpha_c=alpha_c , sort_mode='abs', background= background[i], Show_Option = False)
                a += df_temp["y1"]/IMG_NUM
            df_temp["y1"] = a
            df_temp.rename(columns={'y1': key}, inplace=True)
            df = pd.merge(df, df_temp, on='x')
        df = create_LOR(df)
        print(df.shape)
        createsenPLOT(df,"IMAGENET_SEN"+str(background[i])+'.png',colors=None,markers=markers,line_styles=None,legend=True)
        print("Completed") 
