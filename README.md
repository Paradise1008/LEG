[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


Prerequisites


# LEG

This is the implementaion of the LEG estimator for VGG19


## Usage
Please note that the 
```python
LEG_Explain(vgg_model, image0, filename , noise_lvl , lambda_lvl ,sampling_size, conv)
```
which returns list including the solution, lambda and noise 

```python
generateHeatmap(image0,heatmap,name,style)
```
which is used to visualize LEG


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forksParadise1008/LEG.svg?style=flat-square
[forks-url]: https://github.com/Paradise1008/LEG/network/members
[stars-shield]: https://img.shields.io/github/stars/Paradise1008/LEG.svg?style=flat-square
[stars-url]: https://github.com/Paradise1008/LEG/stargazers
[issues-shield]: https://img.shields.io/github/issues/Paradise1008/LEG.svg?style=flat-square
[issues-url]: https://github.com/Paradise1008/LEG/issues
