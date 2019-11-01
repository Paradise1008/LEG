[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

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

