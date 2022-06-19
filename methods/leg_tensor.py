# Implementaion of LEG/LEG-TV
# Load packages
import keras
import tensorflow as tf

print(f'keras version: {keras.__version__}, tensorflow version:{tf.__version__}')
from keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from keras.applications import vgg19, vgg16, ResNet50, Xception, InceptionV3, inception_v3, xception
from keras.applications.vgg19 import VGG19, decode_predictions, preprocess_input

import cvxpy as cp
import os
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import cv2
import os

from utils import timing_func

# Customized functions
##An example of the predict function of predictions. You can define your own predict functions
##for your own purposes.

class PredictParameter:
    """This class is used to convey the parameter required in LEG for model prediction
    @category: category of prediction
    @value: the predicted probability, [0, 1]
    @opposite_category: if applied, the value represents P(category) - P(opposite_category)
    @show_option: show partial results for debugging
    """

    def __init__(self):
        self.category = -1  # predicted category
        self.value = 0  # predicted probability
        self.show_option = False
        self.opposite_category = -1

    def set_category(self, num):
        """set the value of the specific category,
        -1 means use the category with largest probability"""
        self.category = num

    def hide_show(self):
        """to hide the result of prediction"""
        self.show_option = False

    def set_opposite_category(self, num):
        """set the value of the comparison category"""
        self.opposite_category = num


def predict_MNIST(ori_img, model, pred_paras=PredictParameter()):
    """An Example for VGG16"""
    result_paras = PredictParameter()
    image_input = np.expand_dims(ori_img.copy(), axis=0)
    image_input = image_input / 255.0
    image_input = (image_input - 0.5) * 2
    preds = model.predict(image_input)
    if pred_paras.show_option is True:
        # arg_sort = sorted(range(len(preds[0])), key=lambda k: preds[0,k], inverse=True)
        plt.imshow(ori_img[:, :, 0])
        plt.show()
        arg_sort = np.argsort(-preds[0])
        for j in range(3):
            print(arg_sort[j], preds[0, arg_sort[j]])
    if pred_paras.opposite_category == -1:
        opposite_value = 0
    else:
        opposite_value = preds[0, pred_paras.opposite_category]
    if pred_paras.category == -1:
        temp = np.argmax(preds[0])
        result_paras.category = temp
        result_paras.value = preds[0, temp] - opposite_value
        return result_paras
    result_paras.category = pred_paras.category
    result_paras.value = preds[0, pred_paras.category] - opposite_value
    return result_paras

@timing_func()
def predict_vgg19(ori_img, model, pred_paras=PredictParameter()):
    """An Example for VGG19
    @orig_img: (W, H, C)
    @model: wrapeed model
    preprocessing: normalize image, get predicted category and probability
    """
    result_paras = PredictParameter()
    image_input = np.expand_dims(ori_img.copy(), axis=0)  # [1,224,224,3]
    image_input = vgg19.preprocess_input(image_input)  # normalize?
    preds = model.predict(image_input)  # (1,1000)
    if pred_paras.show_option is True:
        for pred_class in decode_predictions(preds)[0]:
            print(pred_class)
    if pred_paras.category == -1:
        temp = np.argmax(preds[0])
        result_paras.category = temp
        result_paras.value = preds[0, temp]
        return result_paras
    result_paras.category = pred_paras.category
    result_paras.value = preds[0, pred_paras.category]
    return result_paras


def predict_vgg16(ori_img, model, pred_paras=PredictParameter()):
    """An Example for VGG16"""
    result_paras = PredictParameter()
    image_input = np.expand_dims(ori_img.copy(), axis=0)
    image_input = vgg16.preprocess_input(image_input)
    preds = model.predict(image_input)
    if pred_paras.show_option is True:
        for pred_class in decode_predictions(preds)[0]:
            print(pred_class)
    if pred_paras.category == -1:
        temp = np.argmax(preds[0])
        result_paras.category = temp
        result_paras.value = preds[0, temp]
        return result_paras
    result_paras.category = pred_paras.category
    result_paras.value = preds[0, pred_paras.category]
    return result_paras


def predict_resnet(ori_img, model, pred_paras=PredictParameter()):
    """An Example for resnet"""
    result_paras = PredictParameter()
    image_input = np.expand_dims(ori_img.copy(), axis=0)
    image_input = preprocess_input(image_input)
    preds = model.predict(image_input)
    if pred_paras.show_option is True:
        for pred_class in decode_predictions(preds)[0]:
            print(pred_class)
    if pred_paras.category == -1:
        temp = np.argmax(preds[0])
        result_paras.category = temp
        result_paras.value = preds[0, temp]
        return result_paras
    result_paras.category = pred_paras.category
    result_paras.value = preds[0, pred_paras.category]
    return result_paras


def predict_inception(ori_img, model, pred_paras=PredictParameter()):
    """An Example for inception"""
    result_paras = PredictParameter()
    new_img = np.zeros((299, 299, 3))
    dim_img = ori_img.shape
    new_img[0:dim_img[0], 0:dim_img[1], 0:dim_img[2]] = ori_img
    image_input = np.expand_dims(new_img.copy(), axis=0)
    image_input = inception_v3.preprocess_input(image_input)
    preds = model.predict(image_input)
    if pred_paras.show_option is True:
        for pred_class in decode_predictions(preds)[0]:
            print(pred_class)
    if pred_paras.category == -1:
        temp = np.argmax(preds[0])
        result_paras.category = temp
        result_paras.value = preds[0, temp]
        return result_paras
    result_paras.category = pred_paras.category
    result_paras.value = preds[0, pred_paras.category]
    return result_paras


def predict_xception(ori_img, model, pred_paras=PredictParameter()):
    """An Example for inception"""
    result_paras = PredictParameter()
    new_img = np.zeros((299, 299, 3))
    dim_img = ori_img.shape
    new_img[0:dim_img[0], 0:dim_img[1], 0:dim_img[2]] = ori_img
    image_input = np.expand_dims(new_img.copy(), axis=0)
    image_input = xception.preprocess_input(image_input)
    preds = model.predict(image_input)
    if pred_paras.show_option is True:
        for pred_class in decode_predictions(preds)[0]:
            print(pred_class)
    if pred_paras.category == -1:
        temp = np.argmax(preds[0])
        result_paras.category = temp
        result_paras.value = preds[0, temp]
        return result_paras
    result_paras.category = pred_paras.category
    result_paras.value = preds[0, pred_paras.category]
    return result_paras


def import_image(folder_path, num, randomize=False, suffix=".jpeg", seed=409,
                 size=(224, 224), show_option=False):
    """
    import image from folder_path, return a matrix with shape: [n, w, h, c]
    Import_image function by getting the path and target size(**)"""
    import_images = []
    path_list = list(Path(folder_path).rglob('*' + suffix))
    if randomize is True:
        np.random.seed(seed)
        selected = np.random.randint(0, len(path_list), num)
    else:
        selected = np.arange(num)
    for index in selected:
        img = image.load_img(path_list[index], target_size=size)
        if show_option is True:
            print(index)
            plt.imshow(img)
            plt.show()
        img = image.img_to_array(img).astype(int)  # This gives integer from 0-255
        import_images.append(img)
    result = np.stack(import_images, axis=0)
    return result


def rgb2gray(rgb):
    """turn rbg to gray scale
    @rgb: image with channels
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

@timing_func()
def leg_2d_deconv(input_conv, size):
    """ 2d deconvolution function for image with shape [a, b]
    @input_conv: with shape [a, b]
    @size: size of deconvolution, from [a, b] to [a*size, b*size]
    """
    length, width = input_conv.shape  # [28, 28]
    new_length = int(length * size)  # 28*8
    new_width = int(width * size)
    new_input = np.zeros((new_length, new_width))
    for i in range(0, new_length):
        for j in range(0, new_width):
            new_input[i, j] = input_conv[i // size, j // size]
    return new_input

@timing_func()
def leg_conv(input_conv, size, channel):
    """ generalization of function leg_2d_deconv, can be applied to rgb image.
    @channel: the specified channel to apply deconvolution, apply to
    all channels if channel==1.
    Convolution input size from a*b*3 into a/s * b/s *3 and
    compress into one channel if channel=-1"""
    length, width, data_channel = input_conv.shape  # (224, 224, 3)
    new_length = int(length / size)
    new_width = int(width / size)
    new_input = np.zeros((new_length, new_width))  # (28, 28)
    if channel == -1:
        for i in range(0, new_length):
            for j in range(0, new_width):
                new_input[i, j] = np.mean(input_conv[i * size:((i + 1) * size),
                                          j * size:((j + 1) * size), :])
    else:
        for i in range(0, new_length):
            for j in range(0, new_width):
                new_input[i, j] = np.mean(input_conv[i * size:((i + 1) * size),  # ? how does it work?
                                          j * size:((j + 1) * size), channel])
    return new_input

@timing_func()
def create_sigma(matrix_d, is_sparse=True):
    """ create the covariance matrix (sigma_matrix) designed  for the differencing matrix D.
     :param
     matrix_d: the differencing matrix D
     matrix_d.shape = (1624, 784), dplus: D+, pseudo inverse of matrix_d
     """
    p1_p2 = matrix_d.shape[1]
    if is_sparse:  # is_sparse == False by default
        matrix_d = csc_matrix(matrix_d)
        svd_u, svd_s, svd_vh = svds(matrix_d, k=min(p1_p2 - 1, 100))
        sigma = svd_vh.transpose() @ (np.diag(svd_s) ** 2) @ svd_vh + 1 / p1_p2 * np.ones((p1_p2, p1_p2))
        dplus = svd_vh.transpose() @ np.linalg.inv(np.diag(svd_s)) @ svd_u.transpose()
    else:
        svd_u, svd_s, svd_vh = np.linalg.svd(matrix_d, full_matrices=True)
        sigma = svd_vh.transpose() @ np.diag(svd_s ** 2) @ svd_vh + 1 / p1_p2 * np.ones((p1_p2, p1_p2))
        dplus = np.linalg.pinv(matrix_d)  # compute the Moore-Penrose pseudo-inverse of matrix_d:
    # plt.imshow(sigma)
    # plt.show()
    return (dplus, sigma)

@timing_func()
def create_sparse_matrix_d(p_size, padding=False):
    """create the differencing matrix D  in sparse format(***),
    Generate the differencing matrix D with length of the image.
    :param
          p_size: int(img_length / base_size). the length of image. It should not be big, o.w.,the corresponding matrix d will run out of memory.
          padding: boolean to decide whether padding 0 on the border of the image.
    """
    row_num = 0
    if padding is False:
        temp = sparse.lil_matrix((2 * (p_size - 1) * (p_size - 1) + 2 * (p_size - 1), p_size * p_size),
                                 dtype=np.float64)
    else:
        file = Path('Matrix/' + str(p_size) + '/matrix_d_' + str(p_size) + '.npz')  # file:Matrix/28/matrix_d_28.npz
        if file.exists():
            temp = sparse.load_npz(file)
            return temp
        temp = sparse.lil_matrix((2 * (p_size - 1) * (p_size - 1) + 2 * (p_size - 1) + 4 * p_size,
                                  # lil_matrix((M, N), [dtype]) to construct an empty matrix with shape (M, N) dtype is optional, defaulting to dtype='d'.
                                  p_size * p_size), dtype=np.float64)
        for k in range(p_size):  # why?
            temp[row_num, k] = -1
            temp[row_num + 1, p_size * p_size - k - 1] = 1
            temp[row_num + 2, k * p_size] = -1
            temp[row_num + 3, (k + 1) * p_size - 1] = 1
            row_num += 4
    # data = 0?
    for i in range(p_size):
        for j in range(p_size):
            if i == p_size - 1 and j == p_size - 1:
                continue
            if i == p_size - 1:
                row = np.array([row_num, row_num])
                col = np.array([i * p_size + j, i * p_size + j + 1])
                row_num += 1
            elif j == p_size - 1:
                row = np.array([row_num, row_num])
                col = np.array([i * p_size + j, i * p_size + j + p_size])
                data = np.array([1, -1])
                row_num += 1
            else:
                row = np.array([row_num, row_num, row_num + 1, row_num + 1])
                col = np.array([i * p_size + j, i * p_size + j + 1, i * p_size + j, i * p_size + j + p_size])
                data = np.array([1, -1, 1, -1])
                row_num += 2
            temp[row, col] = data
    return temp

@timing_func()
def make_normal_noise(sigma, num_n, cholesky_lmat=None):
    """Create normal perturbatation with mean=0
    @sigma: covariance matrix sigma
    @num_n: number of normal perturbations
    """
    # epsilon = 0.0001
    dim = sigma.shape[0]  # sigma.shape=[784,784]
    cholesky_k = sigma  # + epsilon*np.identity(dim)
    if cholesky_lmat is None:
        cholesky_lmat = np.linalg.cholesky(cholesky_k)  # [784,784],L matrix
    cholesky_u = np.random.normal(loc=0, scale=1, size=dim * num_n).reshape(dim, num_n)  # [784,1000]
    normal_x = np.dot(cholesky_lmat, cholesky_u)  # [784,1000], equal to matmul or @
    return normal_x


# get_mask(ori_img, heatmap, alpha, background=background2, sort_mode='abs')
@timing_func()
def get_mask(ori_img, heatmap, alpha, background, sort_mode):
    """This function is designed to get mask"""
    length = ori_img.shape[0]
    width = ori_img.shape[1]
    result = ori_img.copy()
    heat_flat = heatmap.reshape(1, length * width)
    temp = np.zeros(length * width) + 255
    if sort_mode == 'abs':
        order = np.argsort(-abs(heat_flat))
    else:
        order = np.argsort(-heat_flat)
    temp[order[0, 0:int(alpha * length * width)]] = 0
    temp = temp.reshape(length, width)
    if np.min(background) > 255:
        for i in range(0, length):
            for j in range(0, width):
                if temp[i, j] == 0:
                    result[i, j, :] = result[i, j, :] + (np.min(background) - 255) * (
                            np.random.randint(255, size=(ori_img.shape[2])) - 127)
        return result
    if np.min(background) >= 0:
        for i in range(0, length):
            for j in range(0, width):
                if temp[i, j] == 0:
                    result[i, j, :] = background
        return result
    if np.min(background) < -255:
        for i in range(0, length):
            for j in range(0, width):
                if temp[i, j] == 0:
                    result[i, j, :] = np.random.randint(255, size=(ori_img.shape[2]))
        return result
    for i in range(0, length):
        for j in range(0, width):
            if temp[i, j] == 0:
                if heatmap[i, j] < 0:
                    result[i, j, :] = result[i, j, :] - background
                else:
                    result[i, j, :] = result[i, j, :] + background
    result = np.minimum(np.maximum(result, 0), 255)
    return result

@timing_func()
def sensitivity_anal(predict_func, ori_img, heatmap, model,
                     alpha_c=None, title='sensitivity.jpg', sort_mode='abs',
                     background=None, Show_Option=True, repeat=1):
    """Perform Sensitivity_analysis"""
    if alpha_c is None:
        alpha_c = np.arange(0, 0.8, 0.05)
    if background is None:
        background = ori_img.mean((0, 1))
    num = alpha_c.shape[0]
    ori_data = np.zeros(num)
    if Show_Option:
        if ori_img.shape[2] == 1:
            plt.imshow(ori_img[:, :, 0])
            plt.show()
        else:
            plt.imshow(ori_img)
            plt.show()
        plt.imshow(heatmap)
        plt.show()
    # prediction=?
    # new_img = ?
    # prediction_new = ?
    for i in range(len(alpha_c)):
        if i == 0:
            prediction = predict_func(ori_img, model)
            ori_data[i] = prediction.value
        else:
            for k in range(repeat):
                new_img = get_mask(ori_img=ori_img, heatmap=heatmap, alpha=alpha_c[i],
                                   background=background, sort_mode=sort_mode)
                prediction_new = predict_func(new_img, model, prediction)
                ori_data[i] += prediction_new.value / repeat
            if i == int(len(alpha_c) / 2) and (Show_Option):
                print(new_img.shape)
                if new_img.shape[2] == 1:
                    plt.imshow(new_img[:, :, 0])
                    plt.show()
                else:
                    plt.imshow(new_img)
                    plt.show()
                print(prediction_new.value, prediction_new.category)

    df = pd.DataFrame({'x': alpha_c, 'y1': ori_data})
    if Show_Option:
        plt.plot('x', 'y1', data=df, marker='', color='green', linewidth=2,
                 linestyle='solid', label="leg_ori")
        plt.savefig(title)
        plt.show()
    return df

@timing_func()
def generateHeatmap(image, heatmap, name, style, show_option=True, direction="all"):
    """LEG specialized for ploting heatmap"""
    result_path = "Result"
    dest = os.path.join(result_path, name + '_' + style + '.jpg')
    ht_print = heatmap.copy()
    if direction == "positive":
        ht_print[ht_print < 0] = 0
    if direction == "negative":
        ht_print = -ht_print
        ht_print[ht_print < 0] = 0
    if style == "heatmap_only":
        plt.imshow(ht_print, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.savefig(dest)
    if style == "gray":
        gray = rgb2gray(image / 255.0)
        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.imshow(ht_print, cmap='gist_gray', interpolation='nearest', alpha=0.8)
        plt.colorbar()
        plt.savefig(dest)
    if style == "overlay":
        cam = (heatmap - np.min(heatmap))
        cam = cam / np.max(cam)
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        plt.imshow(np.uint8(cam))
        plt.savefig(dest)
    if show_option:
        plt.show()

@timing_func()
def LEG_perturbation(ori_img, model, predict_func, ori_pred, sigma, num_sample,
                     base_size, conv_size, img_length, img_width, channel_num,
                     cholesky_lmat, gradient_scheme, sample_method, inspect_mode):
    """
    generate summarization of f(x)*x in the paper
    @sample_method: whether to take random sample x1, x2, ..., or take symmetric sample like
    x1, -x1, x2, -x2, ...
    @gradient_scheme: whether to keep the sign of the perturbation like using sum f(x)*x (by default) or
    sum of |f(x)||x|.
    :return:
    """
    perturbation = []
    for channel in range(channel_num):
        model_y = np.zeros((num_sample, 1))  # [1000,1]
        model_x = np.zeros((num_sample, base_size * base_size))  # [1000,784]
        target_y = np.zeros(ori_img.shape)  # [224,224,3]
        if sample_method == 'symmetric':
            normal_noise_1 = make_normal_noise(sigma, int(num_sample / 2), cholesky_lmat)
            normal_noise_2 = -normal_noise_1
            normal_noise = np.concatenate((normal_noise_1, normal_noise_2), axis=1)
        else:
            normal_noise = make_normal_noise(sigma, num_sample, cholesky_lmat)
        for i in range(num_sample):
            perturb_img = np.zeros(ori_img.shape)  # [224,224,3]
            perturb_img[:, :, channel] = leg_2d_deconv(normal_noise[:, i].reshape(base_size, base_size),
                                                       size=conv_size)  # (224, 224)
            perturb_img += ori_img / 255.0  # perturb_img.shape = (224, 224), ori_img.shape = (224, 224, 3)
            # perturb_img = np.minimum(np.maximum(perturb_img, 0), 1)
            perturb_pred = predict_func((perturb_img * 255.0).astype(int), model, ori_pred)
            if gradient_scheme == 'abs':
                pred_diff = abs(perturb_pred.value - ori_pred.value)  # MY IDEA
                rand_new = abs(perturb_img - ori_img / 255.0)  # MY IDEA
            else:
                pred_diff = perturb_pred.value - ori_pred.value
                rand_new = perturb_img - ori_img / 255.0  # just random noise: (224, 224, 3)
            model_y[i, 0] = pred_diff
            model_x[i, :] = leg_conv(rand_new, conv_size, channel).flatten()  # (28* 28).flatten()
            target_y += 1 / num_sample * pred_diff * (rand_new)  # (224, 224, 3)
            if inspect_mode:
                if channel_num == 1:
                    plt.imshow(perturb_img[:, :, 0])
                    plt.show()
                else:
                    plt.imshow(perturb_img)
                    plt.show()
                print("The probability is ", perturb_pred.value, 'with category', perturb_pred.category)
        conv_y = leg_conv(target_y, conv_size, channel).flatten()  # (784,1)
        perturbation.append(
            (model_y, model_x, target_y, conv_y))  # model_y: (1000,1), model_x: (1000,784), target_y:(224, 224, 3)
        if inspect_mode:
            plt.hist(model_y)
            plt.show()
    return perturbation

@timing_func()
def leg_solver(matrix_d, ds_mat, dy_mat, threshold, size, solver='mosek', inspect_mode=True):
    """Given DSmat,DYmat and threshold, calculate LEG.
    Solving the linear programming by mosek solver. A certification from MOSEK is needed.
    @ds_mat: dplus_t*sigma
    @dy_mat: pseudo D * sum(f(x)x)
    @threshold: = lambda_0*dy_mat=lambda*dplus_t*sima. lambda_0 times the absolute maximum of dy_mat
    @solver: optionaal['mosek', 'mosek2']
    """
    # heatmap = ? # might be referenced before assignment
    if solver == 'mosek':
        x0 = cp.Variable(shape=(size * size))
        obj = cp.Minimize(cp.sum(cp.abs(matrix_d * x0)))  # +0.01*cp.sum_squares(matrix_d*x0))
        # It takes more time to solve qudratic problem
        constraint = [cp.max(cp.abs(dy_mat - ds_mat * x0)) <= threshold]
        problem = cp.Problem(obj, constraint)
        problem.solve(verbose=False, solver=cp.MOSEK)
        heatmap = np.array(x0.value).reshape(size, size)
    elif solver == 'mosek2':
        x0 = cp.Variable(shape=(size * size))
        obj = cp.Minimize(cp.sum(cp.abs(matrix_d * x0)) + 0.01 * cp.sum_squares(matrix_d * x0))
        # It takes more time to solve qudratic problem
        constraint = [cp.max(cp.abs(dy_mat - ds_mat * x0)) <= threshold]
        problem = cp.Problem(obj, constraint)
        problem.solve(verbose=False, solver=cp.MOSEK)
        heatmap = np.array(x0.value).reshape(size, size)
    return heatmap

@timing_func()
def LEG_new_perturbation(ori_img, model, predict_func, ori_pred, sigma,
                         num_sample, base_size, conv_size, current_size,
                         img_length, img_width, channel_num, cholesky_lmat,
                         gradient_scheme, sample_method, inspect_mode):
    """ unused function """
    perturbations = []
    img_length, img_width, channel_num = ori_img.shape
    axs = []

    # normal_noise = leg_2d_deconv()
    num_length = int(img_length / (conv_size * current_size))
    num_width = int(img_width / (conv_size * current_size))
    for i in range(num_length):
        for j in range(num_width):
            x_start = i * current_size * conv_size
            x_end = i * current_size * conv_size + current_size * conv_size
            y_start = j * current_size * conv_size
            y_end = j * current_size * conv_size + current_size * conv_size
            target_y = np.zeros((conv_size, conv_size, channel_num))
            for channel in range(channel_num):
                normal_noise = make_normal_noise(sigma, num_sample, cholesky_lmat)
                for sample_id in range(num_sample):
                    perturb_img = np.zeros(ori_img.shape)
                    perturb_img[x_start:x_end, y_start:y_end, channel] = leg_2d_deconv(
                        normal_noise[:, sample_id].reshape(conv_size, conv_size), size=current_size)
                    perturb_img += ori_img / 255.0
                    perturb_img = np.minimum(np.maximum(perturb_img, 0), 1)
                    perturb_pred = predict_func((perturb_img * 255.0).astype(int), model, ori_pred)
                    if inspect_mode:
                        plt.imshow(perturb_img)
                        plt.show()
                        print("The probability is ", perturb_pred.value, 'with category', perturb_pred.category)
                    if gradient_scheme == 'abs':
                        pred_diff = abs(perturb_pred.value - ori_pred.value)  # MY IDEA
                        rand_new = abs(
                            perturb_img[x_start:x_end, y_start:y_end, :] - ori_img[x_start:x_end, y_start:y_end,
                                                                           :] / 255.0)  # MY IDEA
                        rand_new = leg_conv(rand_new, current_size, channel)
                    else:
                        pred_diff = perturb_pred.value - ori_pred.value
                        rand_new = perturb_img[x_start:x_end, y_start:y_end, :] - ori_img[x_start:x_end, y_start:y_end,
                                                                                  :] / 255.0
                        rand_new = leg_conv(rand_new, current_size, channel)
                    # model_y[i, 0] = pred_diff
                    # model_x[i, :] = leg_conv(rand_new, conv_size, channel).flatten()
                    target_y[:, :, channel] += 1 / num_sample * pred_diff * (rand_new)

            perturbations.append(target_y)
            axs.append((x_start, x_end, y_start, y_end))
            # print("we are here", i, j)
    print("we have ", num_length * num_width, " boxes")
    print("The shape of noise is", normal_noise.shape)
    return perturbations, axs

@timing_func()
def LEG_explainer(inputs, model, predict_func, pred_paras=PredictParameter(),
                  base_size=28, conv_size=None, matrix_d=None, penalty=None, sigma=None,
                  noise_lvl=0.02, lambda_arr=None, num_sample=200, method='conv', padding_bol=True,
                  load_matrix_folder=None, gradient_scheme=None, sample_method=None, inspect_mode=False,
                  solver='mosek'):
    """This is the main function to get LEG explanation.
    @conv_size: how much you wish to downsample the image
    @noise_lvl: the noise level for the covariance matrix
    @lambda_arr: lambda value for the TV penality. A large lambda value gives a sparser solution.
    @num_sample: number of the perturbations
    @method: only develop conv. New is not used at all.
    @penalty: If None, LEG is computed; if "TV", LEG-TV is computed
    """
    img_num, img_length, img_width, channel_num = inputs.shape  # input.shape = [1, 224, 224, 3]
    result_list = []
    cholesky_lmat = None
    if conv_size is None:
        conv_size = int(img_length / base_size)  # 224/28=8
    if load_matrix_folder is not None:  # load_matrix_folder = None
        matrix_d = sparse.load_npz(load_matrix_folder + '/matrix_d.npz').toarray()
        dplus = np.load(load_matrix_folder + '/matrix_dplus.npy')
        sigma = np.load(load_matrix_folder + '/matrix_sigma.npy')
        cholesky_lmat = np.load(load_matrix_folder + '/matrix_cholesky.npy')
        print('success')
    else:
        if matrix_d is None:  # base_size:28
            matrix_d = create_sparse_matrix_d(base_size, padding=padding_bol).toarray()  # matrix_d.shape: 1623*783
            print('matrix_d creation completed')
        if sigma is None:
            dplus, sigma_init = create_sigma(matrix_d, is_sparse=False)
            sigma = sigma_init * (noise_lvl ** 2)
            print('Sigma creation complted')
        else:
            dplus = np.linalg.pinv(matrix_d)
    if penalty is None:
        inv_sigma = np.linalg.inv(sigma)
    if lambda_arr is None:
        lambda_arr = [0.05]
    dplus_t = dplus.transpose()
    ds_mat = dplus_t @ sigma
    if method == 'conv':
        for i in range(img_num):
            ori_img = inputs[i]
            ori_pred = predict_func(ori_img, model, pred_paras)
            # if channel_num == 1:
            #    plt.imshow(ori_img[:,:,0])
            #    plt.show()
            # else:
            #    plt.imshow(ori_img)
            #    plt.show()
            print(ori_pred.category, ori_pred.value)
            perturbations = LEG_perturbation(ori_img, model, predict_func, ori_pred, sigma,
                                             num_sample, base_size, conv_size,
                                             img_length, img_width, channel_num, cholesky_lmat,
                                             gradient_scheme, sample_method, inspect_mode)
            # print(perturbations[0][0].shape,perturbations[0][1].shape,perturbations[0][2].shape)
            # inv_sigma = ? might be referenced before assignment
            # temp = ? might be referenced before assignment
            for lambda0 in lambda_arr:
                ht = []
                for perturbation in perturbations:
                    if penalty == 'TV':
                        dy_mat = dplus_t @ perturbation[3]  # dplus_t: 1624,784, perturbation[3], (784,)
                        ref = float(np.max(np.abs(dy_mat)))
                        threshold = ref * lambda0
                        temp = leg_solver(matrix_d, ds_mat, dy_mat, threshold, base_size, solver=solver)
                    elif penalty is None:
                        temp = inv_sigma @ perturbation[3]  # conv_y: (784,)
                        temp = temp.reshape(base_size, base_size)  # (28, 28)
                    ht.append(temp)  # ht.shape: (3, 28, 28)
                    if inspect_mode:
                        print("The heatmap for one channel is")
                        plt.imshow(temp)
                        plt.show()
                if channel_num == 1:
                    # final_ht = leg_2d_deconv(np.asarray(ht[0]), size=conv_size)
                    final_ht = cv2.resize(np.asarray(ht[0]), (base_size * conv_size, base_size * conv_size))
                # print(ht[0].shape)
                else:
                    final_ht = cv2.resize(np.mean(np.abs(np.asarray(ht)), axis=0),
                                          (base_size * conv_size, base_size * conv_size))
                    # final_ht = leg_2d_deconv(np.mean(np.abs(np.asarray(ht)), axis=0), size=conv_size)
                result_list.append((final_ht, ori_img, ori_pred, lambda0))  # final_ht: (224, 224)
    if method == 'new':
        for i in range(img_num):
            ori_img = inputs[i]
            ori_pred = predict_func(ori_img, model, pred_paras)
            current_size = int(min(img_length, img_width) / conv_size)
            final_ht = np.ones((img_length, img_width))

            while current_size > base_size:
                print("current size is", current_size, base_size)

                perturbations, axs = LEG_new_perturbation(ori_img, model, predict_func, ori_pred, sigma,
                                                          num_sample, base_size, conv_size, current_size,
                                                          img_length, img_width, channel_num, cholesky_lmat,
                                                          gradient_scheme, sample_method, inspect_mode)
                # print(len(perturbations),len(axs))
                ht = np.ones((img_length, img_width))
                for perturbation, ax in zip(perturbations, axs):
                    print(perturbation.shape)
                    lambda0 = lambda_arr[0]  # only allow for one lambda
                    temp_ht = []
                    for channel in range(channel_num):
                        if penalty == 'TV':
                            dy_mat = dplus_t @ perturbation[:, :, channel].flatten()
                            ref = float(np.max(np.abs(dy_mat)))
                            threshold = ref * lambda0
                            temp = leg_solver(matrix_d, ds_mat, dy_mat, threshold, conv_size)
                        elif penalty is None:
                            temp = inv_sigma @ perturbation[:, :, channel].flatten()
                            temp = temp.reshape(conv_size, conv_size)
                        temp_ht.append(temp)

                    ht[ax[0]:ax[1], ax[2]:ax[3]] = leg_2d_deconv(np.mean(np.asarray(temp_ht), axis=0),
                                                                 size=current_size)
                final_ht = final_ht * ht
                current_size = int(current_size / conv_size)
            plt.imshow(ori_img)
            plt.show()
            plt.imshow(final_ht)
            plt.show()
        print("Congratulations!")

    return result_list


# test a simple test image
if __name__ == "__main__":
    # setup to use gpu
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    print("We are excuting LEG program", __name__)
    # read the image
    img = image.load_img('/home/zhou/Documents/github/XAI/LEG/Image/yorkie2.jpg', target_size=(224, 224))
    img = image.img_to_array(img).astype(int)  # [224, 224, 3]
    image_input = np.expand_dims(img.copy(), axis=0)  # [1, 224, 224, 3]
    # 123.68, 116.78, 103.94
    image_input = preprocess_input(image_input)
    print("Image has been read successfully")

    # read the model
    # VGG19_MODEL = VGG19(include_top=True)
    VGG19_MODEL = VGG19(include_top=True, weights='imagenet')
    print("VGG19 has been imported successfully")
    # make the prediction of the image by the vgg19
    preds = VGG19_MODEL.predict(image_input)  # <- (n_samples, )
    for pred_class in decode_predictions(preds)[0]:
        print(pred_class)  # ('n04486054', 'triumphal_arch', 0.99987185)
    chosen_class = np.argmax(preds)
    print("The Classfication Category is ", chosen_class)
    begin_time = time()
    # LEG_small = LEG_explainer(np.expand_dims(img.copy(), axis=0), VGG19_MODEL, predict_vgg19, base_size=28,
    # num_sample = 1000, penalty = None)
    LEGTV_small = LEG_explainer(np.expand_dims(img.copy(), axis=0), VGG19_MODEL, predict_vgg19, base_size=28,
                 num_sample = 200, penalty = 'TV', lambda_arr = [0.1, 0.3])
    LEG_large = LEG_explainer(np.expand_dims(img.copy(), axis=0), VGG19_MODEL, predict_vgg19, base_size=28,
                 num_sample = 200, penalty = None)
    LEGTV_large = LEG_explainer(np.expand_dims(img.copy(), axis=0), VGG19_MODEL, predict_vgg19, base_size=28,
                 num_sample = 200, penalty = 'TV', lambda_arr = [0.1, 0.3])
    end_time = time()
    print(f'current wd: {os.getcwd}')
    # np.savetxt("Sample_Test_LEG_small.txt", LEG[0][0])
    # plt.imshow(LEG_small[0][0], cmap='hot', interpolation="nearest")
    # plt.colorbar()
    # plt.savefig("Sample_Test_LEG_small.png")
    plt.imshow(LEGTV_small[0][0], cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_LEGTV0_small.png")
    plt.imshow(LEGTV_small[1][0], cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_LEGTV1_small.png")
    plt.imshow(LEG_large[0][0], cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_LEG_large.png")
    plt.imshow(LEGTV_large[0][0], cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_LEGTV0_large.png")
    plt.imshow(LEGTV_large[1][0], cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.savefig("Sample_Test_LEGTV1_large.png")

    print("LEG computed and saved successfully")
    print("The time spent on LEG explanation is ", round((end_time - begin_time) / 60, 2), "mins")
