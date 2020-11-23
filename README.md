<!--  [![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]-->




# LEG(Linearly Estimated Gradient)
#####v1.0
## Getting Started

### Prerequisites

Please make sure that the following packages are installed.

* [cvxpy](https://github.com/cvxgrp/cvxpy) 
* [Mosek](https://www.mosek.com/documentation/)
* [Keras](https://www.mosek.com/documentation/)
* [matplotlib](https://matplotlib.org/users/installing.html)
* [skimage](https://github.com/scikit-image/scikit-image)

### Documentation

* `methods/LEGv0.py`: Implementation of LEG explainer.
* `ImageNetExp.py`: Create the 500 LEG and LEG-TV explanations on ImageNet.
* `Sanity/`: Implementation of cascading randomizations.

## Usage

The `LEG_explainer` function is called with the following basic inputs:
```python
 LEG_explainer(inputs, model, predict_func, penalty, noise_lvl, lambda_arr, num_sample):
```
The function returns lists for all inputs. Each list contains the saliency map, original image, prediction of original image and corresponding lambda level for saliency map in turn. 

We also provide a customized function for visualization:
```python
generateHeatmap(image, heatmap, name, style, show_option, direction):
```
You can choose the "heatmap_only", "gray" or "overlay" style for the heatmap and decide whether display original saliency or its absolute value by the direction option.

Following is a toy example:
```python
##Import the required packages
from LEG import * 

class Image_Obj:
    def __init__(self,name,Noise,Lambda):
        self.name = name
        self.Noise = Noise
        self.Lambda = Lambda
List = []
List.append(Image_Obj("shark",0.3,0.075))
List.append(Image_Obj("soccer",0.3,0.075))


image_folder = "Image"
vgg_model = vgg19.VGG19(include_top =True)
for i, val in enumerate(List):
    image0 = image.load_img(os.path.join(image_folder,val.name+'.jpg'), target_size=(224,224))
    image0 = image.img_to_array(image0)
    image_input = np.expand_dims(image0.copy(),axis=0)
    image_input = vgg19.preprocess_input(image_input)
    preds = vgg_model.predict(image_input)
    chosen_class = np.argmax(preds)        
    task = LEG_Explain(vgg_model, image0, val.name , np.array([val.Noise]) , np.array([val.Lambda]) ,sampling_size = 200, conv = 8,chosen_class=chosen_class)
    generateHeatmap(image0,task[0].sol,result_path="Result",name = val.name+'_gray.jpg',style = "gray",showOption=True, direction="all")

```
The results are then saved in the `Result` folder.

<!-- <img src="https://github.com/Paradise1008/LEG/blob/master/Result/shark_gray.jpg" width=400 /> <img src="https://github.com/Paradise1008/LEG/blob/master/Result/soccer_gray.jpg" width=400 /> -->

You can modify the content of `List`(an `Image_Obj` class) to get explanations for other images with different parameters. For the choice of parameters, we suggest using [0.02] as the noise level, and using [0.1, 0.3] as lambda values. Plesae note that the change the path for the images for reproducibility.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/Paradise1008/LEG.svg?style=flat-square
[forks-url]: https://github.com/Paradise1008/LEG/network/members
[stars-shield]: https://img.shields.io/github/stars/Paradise1008/LEG.svg?style=flat-square
[stars-url]: https://github.com/Paradise1008/LEG/stargazers
[issues-shield]: https://img.shields.io/github/issues/Paradise1008/LEG.svg?style=flat-square
[issues-url]: https://github.com/Paradise1008/LEG/issues
