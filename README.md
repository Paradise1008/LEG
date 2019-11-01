[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]




# LEG(Linearly Estimated Gradient)

A new saliency estimation framework for explaining black box computer vision models. Here we provide the python implementation of LEG.


## Prerequisites
Here is the list of packages used in LEG. Please make sure they are properly installed before running LEG.
* [cvxpy](https://github.com/cvxgrp/cvxpy) 
* [Mosek](https://www.mosek.com/documentation/)
* [Keras](https://www.mosek.com/documentation/)
* [matplotlib](https://matplotlib.org/users/installing.html)
* [skimage](https://github.com/scikit-image/scikit-image)

## Usage
Please note that the 
```python
LEG_Explain(vgg_model, image0, filename , noise_lvl , lambda_lvl ,sampling_size, conv)
```
which returns list including the solution, lambda and noise 


We also provide a customized function for heatmap visualization.
```python
generateHeatmap(image0,heatmap,name,style)
```

Here is a toy example:


You can modify the class object in order to get your own interpretation


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/Paradise1008/LEG.svg?style=flat-square
[forks-url]: https://github.com/Paradise1008/LEG/network/members
[stars-shield]: https://img.shields.io/github/stars/Paradise1008/LEG.svg?style=flat-square
[stars-url]: https://github.com/Paradise1008/LEG/stargazers
[issues-shield]: https://img.shields.io/github/issues/Paradise1008/LEG.svg?style=flat-square
[issues-url]: https://github.com/Paradise1008/LEG/issues
