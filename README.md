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

```bash
conda create --no-default-packages -n pyroothair python # create fresh conda environment
conda activate pyroothair # activate environment
conda env config vars set nnUNet_raw='~/nnUNet_raw' nnUNet_preprocessed='~/nnUNet_preprocessed' nnUNet_results='~/nnUNet_results' # optional: set nnUNet environment variables to conda env to avoid warning when running pyRootHair. Ignore if you already have nnUNet paths setup

conda deactivate pyroothair
conda activate pyroothair
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyroothair
```  
After installation, run `pyroothair`. You should be greeted with this output:

```
#########################################
     Thank you for using pyRootHair!     
#########################################


...No GPU Detected...

usage: pyRootHair [-h] [-i [IMG_DIR]] [-b [BATCH_ID]] [-p [{1,2,3}]] [--resolution [HEIGHT_BIN_SIZE]]
                  [--conv [CONV]] [--frac [FRAC]] [-o SAVE_PATH] [--plot_segmentation] [--plot_transformation]
                  [--plot_summary] [--override_model_path OVERRIDE_MODEL_PATH]
                  [--override_model_checkpoint OVERRIDE_MODEL_CHKPOINT] [--rfc_model_path RFC_MODEL_PATH]
                  [--sigma_min SIGMA_MIN] [--sigma_max SIGMA_MAX] [--input_mask [INPUT_MASK]]
pyRootHair: error: The following arguments are required when running pyRootHair using the main pipeline: ['-i/--input', '-b/--batch_id']
```

## How to use pyRootHair

Here is a breakdown of the basic required arguments for running pyRootHair with the default segmentation pipeline:

### Default Pipeline

The default segmentation pipeline in pyRootHair uses a CNN to perform image segmentation. As such, a GPU is almost certainly required to maximize segmentation speed and performance. However, it is still possible to run the default segmentation pipeline without a GPU, if you do not have access to one. Segmentation performance will be **extremely** slow when using a CPU, and will very likely produce out-of-memory crashes unless your images are very small in size.

The following arguments are required to run the standard segmentation pipeline:

```bash
-i/--input: the filepath to the directory containing the images you want to process  
-b/--batch_id: a unique ID associated with each batch of images you are processing per run. Can be species/genotype name, or date, or anything that is easily identifiable for you.
```
Request a GPU on your cluster. On a SLURM system, this requests a single GPU with 30GB VRAM:

```bash
srsh --partition=gpu --gpus=1 --mem=30G
```

To verify a GPU has been correctly requested, run `nvidia-smi`. You should get some information about the GPU printed to your screen.

To run pyroothair:

```bash
pyroothair -i ~/Images/Wheat/soissons/ -b Soissons
```

For each batch of input images, the images are renamed with a suffix, and saved in a new directory called 'adjusted_images'. 

In the above example, the renamed images will be stored in `~/images/wheat/adjusted_images/Soissons`, and segmentation masks will be stored in `~/images/wheat/masks/Soissons`. 

Please refer to [this] (https://github.com/iantsang779/pyRootHair/blob/main/workflow.md#extracting-traits-from-the-root-hair-mask) section for available flags and arguments.

To view help documentation on each argument: `pyroothair -h`

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
**`--conv`**: [option] conversion factor to translate pixel data to mm. Since you can only adjust the value of **`--conv`** once per run, you must only run pyRootHair on a batch of images that has been captured at the same magnification! If you have different images caputed at different magnification/zoom distances, you will need to split them into separate batches, and manually adjust the value for **`--conv`**.  
**`--frac`**: [option] control the degree of LOWESS smoothing of lines for average RHL, RHD, and individual segment RHL and RHD. Larger values will increase smoothing factor, while smaller values decrease the smoothing factor of the line. See [this](https://github.com/iantsang779/pyRootHair/blob/main/workflow.md#summary-plots) for a visual representation of the regression lines. Value must be between 0 and 1!  


## Dependencies
- python: 3.12.3
- numpy: 2.0.2
- pandas: 2.2.3
- matplotlib: 3.9.0
- imageio: 2.36.1
- scikit-image: 0.24.0
- scipy: 1.14.1
- statsmodels: 0.14.4
- scikit-learn: 1.5.2
- python-magic: 0.4.27
- nnUNetv2: 2.5.1
- torch: 2.5.1


## pyRootHair Workflow

If you are interested in learning how pyRootHair works behind the scenes, please check out [this] (https://github.com/iantsang779/pyRootHair/blob/main/workflow.md) in-depth walk through.


## Generating Binary Masks
pyRootHair will accept binary masks of any images as long as they are arrays of 0s, 1s and 2s. It is reccommended that you use ilastik (https://www.ilastik.org/) to generate masks, as it is simple and requires minimal expertise.

This section is not a tutorial on how to use ilastik, rather, a demonstration on what the masks need to look like if you wish to generate your own masks suitable for pyroothair.

1.) Under the `1. Input Data` tab, upload your raw image(s). Ensure that the input image only has 3 channels! 
2.) Select all features under the `Feature Selection` tab.  
3.) Specify the following label categories. The label order **must** be as shown!  

![alt text](demo/ilastik_classes.png)

4.) The root hairs must wrap around the root, especially at the root tip:

![alt text](demo/demo_mask.png)

5.) After generating the mask, select `Source: Simple Segmentation` under `4. Prediction Export`. Click on `Choose Export Image Settings`, and set the output file format to `.png`, then hit `Export All`. 

6.) Once generated, the mask should be converted such that each pixel is a 0, 1 or 2 in the array. By default, ilastik saves each pixel associated with the background as 1, root hair as 2, and root as 3. To convert the mask:

`pyroothair_convert_mask -i path/to/your/generated/mask`

You should see a the following message if the conversion has been successful: `...Saved converted input mask XXX in ZZZ...`. You can now use this mask to either train a random forest model, or process as a standalone mask.


















