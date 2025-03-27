# pyRootHair

Welcome to the pyRootHair github repository! Here, you will find all necessary information on how to install and setup pyRootHair, detailed information about the various pipelines and options available, and an in-depth tutorial on how pyRootHair works.

Please do not hesitate to submit a pull-request, or get in touch via [email](ian.tsang@niab.com) if you have any questions, suggestions or concerns!


## Installation instructions

`mkdir ~/pyroothair`  
`git clone https://github.com/iantsang779/pyRootHair`  
`cd pyroothair`  

## How to use pyRootHair

WIP.

## An in-depth dive into the pyRootHair workflow

This section is for users that are curious about how pyRootHair extracts traits from a given input image. In essence, this is a step by step process of the functions and logic within the source code. For this demonstration, I will be using the demo image `karim_demo.png` and the corresponding binary segmentation mask `karim_demo_mask.png`.

#### Loading the necessary libraries

First, we import the required libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio

from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.transform import rotate, warp, PiecewiseAffineTransform
from scipy.ndimage import convolve
from scipy.spatial.distance import euclidean
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
```

For reference, the following versions were used at the time of writing this tutorial:

- python: 3.12.3
- numpy: 2.0.2
- pandas: 2.2.3
- matplotlib: 3.9.0
- scikit-image: 0.24.0
- scipy: 1.14.1
- statsmodels: 0.14.4


