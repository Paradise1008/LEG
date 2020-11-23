###Make the MNIST dataset sensitivity ###
# Load packages
import sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
sys.path.append('../')
from methods.LEGv0 import *
from tensorflow.keras.models import load_model
from keras.datasets import mnist
import keras
from keras import backend as K
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


def create_grid_SenPlot(dfs,filename,colors=None,markers=None,line_styles=None, titles=None,nrow=2, ncol=2):
    if colors is None:
        colors = ['tab:red','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    if markers is None:
        markers = ['D','*','o','v','^','s','*','+']
    if line_styles is None:
        line_styles = ['-','-','--','--','dotted','-.','solid',(0, (3, 1, 1, 1))]
    if titles is None:
        titles = ['(a)','(b)','(c)','(d)','(e)','(f)']

    plt.style.use('seaborn-whitegrid')
    my_dpi=400
    plt.figure()
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rc('font', size=20)          # controls default text sizes
    #plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    #plt.rc('axes', titlesize=16)     # fontsize of the axes title
    #plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
   # plt.rc('legend', fontsize=16)    # legend fontsize
    #plt.rc('figure', titlesize=40)  # fontsize of the figure title
    col_names = dfs[0].columns.values
    x_axis = col_names[0]
    methods = col_names[1:]
    fig, axs = plt.subplots(nrow, ncol,figsize=(30,5))
    if nrow == 1:
        for j in range(ncol):
            df = dfs[j]
            for k in range(len(methods)):
                axs[j].plot(df[x_axis], df[methods[k]], linestyle= line_styles[k], marker=markers[k], color=colors[k], linewidth=1,label=methods[k])
            axs[j].set_title(titles[j],size=24)
    elif ncol == 1:
        for i in range(nrow):
            df = dfs[i]
            for k in range(len(methods)):
                axs[i].plot(df[x_axis], df[methods[k]], linestyle= line_styles[k], marker=markers[k], color=colors[k], linewidth=1,label=methods[k])
            axs[i].set_title(titles[i],size=24)
    else:
        for i in range(nrow):
            for j in range(ncol):
                df = dfs[i*ncol+j]
                for k in range(len(methods)):
                    axs[i, j].plot(df[x_axis], df[methods[k]], linestyle= line_styles[k], marker=markers[k], color=colors[k], linewidth=1,label=methods[k])
                axs[i, j].set_title(titles[i*ncol+j],size=24)
    for ax in axs.flat:
        ax.set(xlabel='Perturbation Size', ylabel='Log Odds Ratio')

    #for ax in axs.flat:
    #    ax.label_outer()
    plt.legend(title="METHODS",loc='upper center', bbox_to_anchor=(-1.3, -0.1),
          fancybox=True, shadow=True, ncol=7,fontsize='x-large')
    #plt.legend( title="METHODS",loc=1, fontsize='small',bbox_to_anchor=(1.65, 2.2),fancybox=True, shadow=True)
    plt.savefig(filename, bbox_inches="tight")
    return "Completed"

if __name__ == "__main__":
    MNIST_MODEL = load_model("/CCAS/home/shine_lsy/LEG/mnist/model/MNIST_new_model_py.h5")
    print("Model imported")
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

    folder_path = '/CCAS/home/shine_lsy/LEG/mnist/result/'
    IMG_NUM = 100
    sa_leg = np.zeros((IMG_NUM,28,28))
    sa_legtv = np.zeros((IMG_NUM,28,28))
    sa_cshap = np.zeros((IMG_NUM,28,28))
    sa_lime = np.zeros((IMG_NUM,28,28))
    sa_gradcam = np.zeros((IMG_NUM,28,28))
    sa_kshap = np.zeros((IMG_NUM,28,28))
    sa_sal = np.zeros((IMG_NUM,28,28))
    for i in range(IMG_NUM):
        sa_sal[i] = np.loadtxt(folder_path+'SALIENCY/'+str(i)+'.txt')
        sa_gradcam[i] = np.loadtxt(folder_path+'GradCam/'+str(i)+'.txt')
        sa_kshap[i] = np.loadtxt(folder_path+'KSHAP/'+str(i)+'.txt')
        sa_cshap[i] = np.loadtxt(folder_path+'CSHAP4/'+str(i)+'.txt')
        sa_leg[i] = np.loadtxt(folder_path+'LEG/'+str(i)+'.txt')
        sa_legtv[i] = np.loadtxt(folder_path+'LEGTV/'+str(i)+'.txt')
        sa_lime[i] = np.loadtxt(folder_path+'LIME/'+str(i)+'.txt')
    print("Explanations Imported")
    METHODS = {'LEG':sa_leg , 'LEG-TV':sa_legtv, 'saliency':sa_sal  , 'C-Shap':sa_cshap , 'LIME':sa_lime, 'GradCam': sa_gradcam, 'KernelSHAP': sa_kshap}
    INDEX = np.arange(IMG_NUM)
    INPUTS = x_test[INDEX]
    alpha_c = np.linspace(0,0.5,100)
    PLOT_DATA = []
    background = [-100,-254,None,-1000]
    for i in range(len(background)):
        print("sensitivity ananlysis is computing "+str(i)+"/"+str(len(background)))
        df=pd.DataFrame({'x' : alpha_c})
        a=np.zeros(len(alpha_c))
        for key in METHODS:
            a=np.zeros(len(alpha_c))
            for k in range(IMG_NUM):
                ori_img = INPUTS[k]
                heatmap = METHODS[key][INDEX[k]]
                #print(key)
                if background[i] == -1000:
                    df_temp = sensitivity_anal(predict_MNIST , ori_img, heatmap, MNIST_MODEL, alpha_c=alpha_c , sort_mode='abs', background= background[i], Show_Option = False, repeat=2)
                else: 
                    df_temp = sensitivity_anal(predict_MNIST , ori_img, heatmap, MNIST_MODEL, alpha_c=alpha_c , sort_mode='abs', background= background[i], Show_Option = False)
                a += df_temp["y1"]/IMG_NUM
            df_temp["y1"] = a
            df_temp.rename(columns={'y1': key}, inplace=True)
            df = pd.merge(df, df_temp, on='x')
        df = create_LOR(df)
        PLOT_DATA.append(df)
    markers = ['']*10
    create_grid_SenPlot(PLOT_DATA,'NEW_MNIST_SEN.eps',colors=None,markers=markers,line_styles=None, titles=None,nrow=1, ncol=len(background))
    print("Completed")

