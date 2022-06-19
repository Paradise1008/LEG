# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:52:45 2020

@author: luosh
"""

# Load packages
import sys
import keras

sys.path.append('../')
from ..methods.LEGv0 import *
# from Plots.Plots import *
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Customized functions
def create_grid_SenPlot(dfs, filename, colors=None, markers=None, line_styles=None, titles=None, nrow=2, ncol=2,
                        my_dpi=400):
    if colors is None:
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                  'tab:cyan']
    if markers is None:
        markers = ['D', '*', 'o', 'v', '^', 's', '*', '+']
    if line_styles is None:
        line_styles = ['-', '-', '--', '--', 'dotted', '-.', 'solid', (0, (3, 1, 1, 1))]
    if titles is None:
        titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    plt.style.use('seaborn-whitegrid')
    plt.figure(dpi=my_dpi)
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    col_names = dfs[0].columns.values
    x_axis = col_names[0]
    methods = col_names[1:]
    fig, axs = plt.subplots(nrow, ncol)
    if nrow == 1:
        for j in range(ncol):
            df = dfs[j]
            for k in range(len(methods)):
                axs[j].plot(df[x_axis], df[methods[k]], linestyle=line_styles[k], marker=markers[k], color=colors[k],
                            linewidth=1, label=methods[k])
            axs[j].set_title(titles[j])
    elif ncol == 1:
        for i in range(nrow):
            df = dfs[i]
            for k in range(len(methods)):
                axs[i].plot(df[x_axis], df[methods[k]], linestyle=line_styles[k], marker=markers[k], color=colors[k],
                            linewidth=1, label=methods[k])
            axs[i].set_title(titles[i])
    else:
        for i in range(nrow):
            for j in range(ncol):
                df = dfs[i * ncol + j]
                for k in range(len(methods)):
                    axs[i, j].plot(df[x_axis], df[methods[k]], linestyle=line_styles[k], marker=markers[k],
                                   color=colors[k], linewidth=1, label=methods[k])
                axs[i, j].set_title(titles[i * ncol + j])
    for ax in axs.flat:
        ax.set(xlabel='Perturbation Size', ylabel='Log Odds Ratio')
    for ax in axs.flat:
        ax.label_outer()
    plt.legend(title="METHODS", loc=1, fontsize='small', bbox_to_anchor=(1.65, 2.2), fancybox=True, shadow=True)
    plt.savefig(filename, bbox_inches="tight")
    return "Completed"


if __name__ == "__main__":
    # Plot 1
    MNIST_MODEL = load_model("C:/Users/luosh/Desktop/CVPR-LEG-CODE/MNIST/model/MNIST_model_py.h5")
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

    folder_path = 'C:/Users/luosh/Desktop/LEG/MNISTexperiment/Explanations/'
    IMG_NUM = 100
    sa_sal = np.zeros((IMG_NUM, 28, 28))
    sa_dl = np.zeros((IMG_NUM, 28, 28))
    sa_elrp = np.zeros((IMG_NUM, 28, 28))
    sa_sv = np.zeros((IMG_NUM, 28, 28))
    sa_leg = np.zeros((IMG_NUM, 28, 28))
    sa_leg_tv = np.zeros((IMG_NUM, 28, 28))
    sa_occ = np.zeros((IMG_NUM, 28, 28))
    for i in range(IMG_NUM):
        sa_sal[i] = np.loadtxt(folder_path + 'SALIENCY/' + str(i) + '.txt')
        sa_dl[i] = np.loadtxt(folder_path + 'DEEPLIFT/' + str(i) + '.txt')
        sa_elrp[i] = np.loadtxt(folder_path + 'ELRP/' + str(i) + '.txt')
        sa_sv[i] = np.loadtxt(folder_path + 'SHAP/' + str(i) + '.txt')
        sa_leg[i] = np.loadtxt(folder_path + 'NEW_LEG/' + str(i) + '.txt')
        sa_leg_tv[i] = np.loadtxt(folder_path + 'NEW_LEGTV/' + str(i) + '.txt')
        sa_occ[i] = np.loadtxt(folder_path + 'OCC/' + str(i) + '.txt')

    METHODS = {'LEG': sa_leg, 'LEGTV': sa_leg_tv, 'saliency': sa_sal, 'deeplift': sa_dl, 'elrp': sa_elrp, 'occ': sa_occ,
               'sv': sa_sv}
    INDEX = np.arange(IMG_NUM)
    INPUTS = x_test[INDEX]
    alpha_c = np.linspace(0, 0.2, 10)
    PLOT_DATA = []
    background = [-50, -254, None, -1000]
    for i in range(4):
        df = pd.DataFrame({'x': alpha_c})
        a = np.zeros(len(alpha_c))
        for key in METHODS:
            a = np.zeros(len(alpha_c))
            for k in range(IMG_NUM):
                ori_img = INPUTS[k]
                heatmap = METHODS[key][INDEX[k]]
                print(key)
                df_temp = sensitivity_anal(predict_MNIST, ori_img, heatmap, MNIST_MODEL, alpha_c=alpha_c,
                                           sort_mode='abs', background=background[i], Show_Option=False)
                a += df_temp["y1"] / IMG_NUM
            df_temp["y1"] = a
            df_temp.rename(columns={'y1': key}, inplace=True)
            df = pd.merge(df, df_temp, on='x')
        df = create_LOR(df)
        PLOT_DATA.append(df)
    markers = [''] * 10
    create_grid_SenPlot(PLOT_DATA, 'MNIST_SEN.pdf', colors=None, markers=markers, line_styles=None, titles=None, nrow=1,
                        ncol=4)
