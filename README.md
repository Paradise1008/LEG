# LEG

This is the implementaion of the LEG estimator for VGG19

## Installation

Need to install Mosek to run this program. You may find how to install it .[here].(https://www.mosek.com/)

## Usage

```python
LEG_Explain(vgg_model, image0, filename , noise_lvl , lambda_lvl ,sampling_size, conv)
```
which returns list including the solution, lambda and noise 

```python
generateHeatmap(image0,heatmap,name,style)
```
which is used to visualize LEG

