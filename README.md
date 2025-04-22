# pyRootHair

Welcome to the pyRootHair github repository! 

Here, you will find all necessary information on how to install and setup pyRootHair, detailed information about the various pipelines and options available, and an in-depth tutorial on how pyRootHair works.

Please do not hesitate to submit a pull-request, or get in touch via email if you have any questions, suggestions or concerns!


## Table of Contents
  - [Installation instructions](#installation-instructions)
  - [How to use pyRootHair](#how-to-use-pyroothair)
    - [GPU and CPU Arguments](#gpu-and-cpu-arguments)
    - [Plotting Arguments (Optional)](#plotting-arguments-optional)
    - [Data Arguments (Optional)](#data-arguments-optional)
    - [Alternative Pipeline Options (Optional)](#alternative-pipeline-options-optional)
  - [Dependencies](#dependencies)
  - [pyRootHair Workflow](#pyroothair-workflow)


## Installation instructions

ADD DOCKER/CONDA STUFF HERE 
`mkdir ~/pyroothair`  
`git clone https://github.com/iantsang779/pyRootHair`  
`cd pyroothair`  

## How to use pyRootHair

Here is a breakdown of the basic required arguments for running pyRootHair with the default segmentation pipeline:

### Default Pipeline

The default segmentation pipeline in pyRootHair uses a CNN to perform image segmentation. As such, a GPU is almost certainly required to maximize segmentation speed and performance. However, it is still possible to run the default segmentation pipeline without a GPU, if you do not have access to one. Segmentation performance will be **extremely** slow when using a CPU, and will very likely produce out-of-memory crashes unless your images are very small in size.

#### GPU and CPU Arguments

The following arguments are required to run the standard segmentation pipeline, with or without a GPU. 

```bash
-i/--input: the filepath to the directory containing the images you want to process  
-b/--batch_id: a unique ID associated with each batch of images you are processing per run. Can be species/genotype name, or date, or anything that is easily identifiable for you.
```
An example of a basic command is as follows:

```bash
python main.py -i ~/Images/Wheat/soissons/ -b soissons
```

### Random Forest Pipeline

If you do not have access to a GPU and desire better segmentation performance, it is possible to train your own random forest segmentation model, and ask pyRootHair to use that model instead.

The random forest segmentation model is nowhere near as powerful as the CNN, and as such, will struggle with image anomalies or noise. Please treat this as an experimental feature, and use with caution.

You will need to ensure that all the images are relatively consistent in terms of lighting, appearance, root hair morphology, and have the same input dimensions. Should your images vary for these traits, you will need to train separate random forest models for different batches of images.

To train a random forest model, you will need to train the model on a single representative example of an image, and a corresponding binary mask of the image. See [this]() section for details on how to generate suitable binary masks.



### Plotting Arguments (Optional)

**`--output`**: [option] specify a filepath to store output data tables and plots (if plotting flags are enabled).  
**`--plot_segmentation`**: [flag] toggle plotting of predicted binary masks for each image (straightened mask, root hair segments, and cropped root hair segments). Must provide a valid filepath for **`--output`**.  
**`--plot_summary`**: [flag] toggle plotting of summary plots describing RHL and RHD for each image. Must provide a valid filepath for `--output`.  
**`--plot_transformation`**: [flag] toggle plotting of co-ordinates illustrating how each input image is warped and straightened. Useful for debugging any strangely warped masks. Must provide a valid filepath for **`--output`**.  

### Data Arguments (Optional)
**`--resolution`**: [option] change bin size (in pixels) for sliding window down each root hair segment. See [this](https://github.com/iantsang779/pyRootHair/blob/main/workflow.md#extracting-traits-from-the-root-hair-mask) section for more details.  
**`--split_segments`**: [option] change the rectangular area of pixels to set as 'False' around the located root tip to ensure separation of the root hair mask into 2 sections, left and right. By default, the boundary of the rectangular area is 20px (width) by 60px (height). By default, the height of the rectangle is calculated by multiplying the user input/default value by 3 to ensure any thick root hair mask section around the root tip can still be split.  
**`--rhd_filt`**: [option] change the threshold (in mm) used to filter out small RHL values calculated along the root. The predicted binary masks for each image can sometimes predict a moderately thick section of root hair around the root tip and regions where there are no visible root hairs. As such, you can manually adjust this value to ensure that bald sections of roots have 0 RHL!  
**`--rhd_filt`**: [option] change the threshold (in mm$^{2}$) used to filter out small RHD values calculated along the root. Same as **`--rhd_filt`**.  
**`--conv`**: [option] conversion factor to translate pixel data to mm. Since you can only adjust the value of **`--conv`** once per run, you must only run pyRootHair on a batch of images that has been captured at the same magnification! If you have different images caputed at different magnification/zoom distances, you will need to split them into separate batches, and manually adjust the value for **`--conv`**.  
**`--frac**`: [option] control the degree of LOWESS smoothing of lines for average RHL, RHD, and individual segment RHL and RHD. Larger values will increase smoothing factor, while smaller values decrease the smoothing factor of the line. See [this](https://github.com/iantsang779/pyRootHair/blob/main/workflow.md#summary-plots) for a visual representation of the regression lines. Value must be between 0 and 1!  




## Dependencies

- python: 3.12.3
- numpy: 2.0.2
- pandas: 2.2.3
- matplotlib: 3.9.0
- imageiov3: 2.36.1
- scikit-image: 0.24.0
- scipy: 1.14.1
- statsmodels: 0.14.4
- sklearn: 1.5.2
- nnUNetv2: 2.5.1
- torch: 2.5.1

You will need a GPU with at least 20GB VRAM to utilize pyRootHair to it's fullest extent. 


## pyRootHair Workflow

If you are interested in learning how pyRootHair works behind the scenes, please check out [this] (https://github.com/iantsang779/pyRootHair/blob/main/workflow.md) in-depth walk through.



## Generating Binary Masks

<<<<<<< HEAD
pyRootHair will accept binary masks of any images as long as they are arrays of 0s, 1s and 2s.
=======

>>>>>>> 3ad096fa3829ad351136a7bbc5e8e185a4ea4dad





















