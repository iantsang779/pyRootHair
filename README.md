# pyRootHair

Welcome to the pyRootHair github repository! 

Here, you will find all necessary information on how to install and setup pyRootHair, detailed information about the various pipelines and options available, and an in-depth tutorial on how pyRootHair works.

Please do not hesitate to submit a pull-request, or get in touch via email if you have any questions, suggestions or concerns!


## Table of Contents
- [pyRootHair](#pyroothair)
  - [Table of Contents](#table-of-contents)
  - [Installation instructions](#installation-instructions)
  - [User Guide](#user-guide)
    - [Default Pipeline](#default-pipeline)
      - [Flags/Arguments](#flagsarguments)
        - [`-i/--input` (REQUIRED - ARGUMENT - STRING)](#-i--input-required---argument---string)
        - [`-o/--output` (REQUIRED - ARGUMENT - STRING)](#-o--output-required---argument---string)
        - [`--batch_id/-b` (REQUIRED - ARGUMENT - STRING/INT/FLOAT)](#--batch_id-b-required---argument---stringintfloat)
        - [`--conv` (OPTIONAL - ARGUMENT - INT)](#--conv-optional---argument---int)
        - [`--resolution` (OPTIONAL - ARGUMENT - INT)](#--resolution-optional---argument---int)
        - [`--frac` (OPTIONAL - ARGUMENT - FLOAT)](#--frac-optional---argument---float)
        - [`--plot_segmentation` (OPTIONAL - FLAG)](#--plot_segmentation-optional---flag)
        - [`--plot_transformation` (OPTIONAL - FLAG)](#--plot_transformation-optional---flag)
        - [`--plot_summary` (OPTIONAL - FLAG)](#--plot_summary-optional---flag)
        - [`-p/--pipeline` (OPTIONAL - ARGUMENT - STR)](#-p--pipeline-optional---argument---str)
    - [Random Forest Pipeline](#random-forest-pipeline)
      - [Training the Random Forest Model](#training-the-random-forest-model)
      - [Deploying the Random Forest Model](#deploying-the-random-forest-model)
    - [Single Mask Pipeline](#single-mask-pipeline)
  - [Generating Binary Masks](#generating-binary-masks)
  - [pyRootHair Workflow](#pyroothair-workflow)



## Installation instructions

```bash
conda create --no-default-packages -n pyroothair python # create fresh conda environment
conda activate pyroothair # activate environment
conda env config vars set nnUNet_raw='~/nnUNet_raw' nnUNet_preprocessed='~/nnUNet_preprocessed' nnUNet_results='~/nnUNet_results' # optional: set nnUNet environment variables to conda env to avoid warning when **running** pyRootHair. Ignore if you already have nnUNet paths setup

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

usage: pyRootHair [-h] [-i [IMG_DIR]] [-b [BATCH_ID]] [-p [{cnn,random_forest,single}]] [--resolution [HEIGHT_BIN_SIZE]]
                  [--conv [CONV]] [--frac [FRAC]] [-o SAVE_PATH] [--plot_segmentation] [--plot_transformation]
                  [--plot_summary] [--rfc_model_path RFC_MODEL_PATH]
                  [--sigma_min SIGMA_MIN] [--sigma_max SIGMA_MAX] [--input_mask [INPUT_MASK]]
pyRootHair: error: The following arguments are required when running pyRootHair using the main pipeline: ['-i/--input', '-b/--batch_id']
```

## User Guide
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

A basic command to run pyRootHair is as follows:

```bash
pyroothair -i ~/Images/Wheat/Soissons/ -b Soissons -o ~/Output/Soissons
```
#### Flags/Arguments

To view options/help messages for each flag, enter `pyroothair -h`

##### `-i/--input` (REQUIRED - ARGUMENT - STRING)
Filepath containing your input images. You can split your images into folders depending on what makes sense for your inputs. Images can be split by genotype, species, condition, treatment, timestamp etc.

##### `-o/--output` (REQUIRED - ARGUMENT - STRING)
Filepath to store outputs. By default, only the raw and summary data tables will be saved to this path. Any additional outputs (e.g. with `--plot_segmentation`) will be stored here as well.

##### `--batch_id/-b` (REQUIRED - ARGUMENT - STRING/INT/FLOAT)
In the above example, the renamed images will be stored in `~/images/wheat/adjusted_images/Soissons`, and segmentation masks will be stored in `~/images/wheat/masks/Soissons`. The `--batch_id/-b` argument assigns a unique ID to the entire batch of images given by `-i`. This could be an ID for a particular genotype (e.g. Soissons, a wheat variety), or a timestamp (e.g. each batch of images are from a specific timepoint). You must assign a unique ID for each run of new images!

##### `--conv` (OPTIONAL - ARGUMENT - INT)
You must ensure that all input images for each batch were taken using at the same magnification setting. You will need to adjust the pixel to mm conversion factor for your input images, which you can determine from measuring a scale bar on your images using the FIJI (ImageJ) 'Analyze' > 'Set Scale' option. You must set the number of pixels per mm using `--conv` each time you run pyRootHair. If you have images taken at different magnification settings, you will need to split them into separate batches, and manually adjust the value of `--conv`.

##### `--resolution` (OPTIONAL - ARGUMENT - INT)
Your input images can be of different shapes, as long as they are relatively consistent in size and have only 3 channels (R,G,B). pyRootHair computes a sliding window down the root, and takes measurement from bins. Using `--resolution`, you can tweak the bin size (in pixels) of the sliding window. For example, if your input images have the shape 800 (width) x 1500 (height), there will be 75 data points ($\frac{1500}{20} = 75$) for RHL and RHD for each side root hair segment using the default `--resolution` value of 20 pixels. 

##### `--frac` (OPTIONAL - ARGUMENT - FLOAT)
Controls the degree of LOWESS smoothing for the lines used to model average RHL and RHD for each image. Since measurements from each bin in the sliding window is noisy, a smoothed line over these points reduces the effect of variation between bin measurements. A smaller value for `--frac` decreases the smoothing effect, i.e. the line will better fit the RHL/RHD data for each bin, but will fluctuate significantly. A larger value for `--frac` increases the smoothing effect, i.e the line will be much smoother through the RHL/RHD data for each bin, but be a worse fit. See [this] (https://github.com/iantsang779/pyRootHair/blob/main/workflow.md#summary-plots) for a visual representation of the regression lines. Value must be a floating point number (e.g. 0.15) between 0 and 1. The default value is recommended. 

##### `--plot_segmentation` (OPTIONAL - FLAG)
Toggle plotting of segmented masks for each image. For each input image, `--plot_segmentation` saves the straightened mask, a mask of just the root hair segments, and the cropped root hair segments. Masks are saved in filepath specified in `--output`

##### `--plot_transformation` (OPTIONAL - FLAG)
Toggle plotting of co-ordinates illustrating how each root is warped and straightened. Can be helpful to check if an image has been poorly warped. Plots are saved in filepath specified in `--output`

##### `--plot_summary` (OPTIONAL - FLAG)
Toggle plotting of summary plots describing RHL and RHD for each input image. Plots are saved in filepath specified in `--output`

##### `-p/--pipeline` (OPTIONAL - ARGUMENT - STR)
Specify which pyRootHair pipeline to run. By default, the main pipeline (`-p cnn`) uses a CNN to segment input images, with or without a GPU (GPU preferred, of course!). If you wish to use a random forest model instead to perform image segmentation, you must specify `-p random_forest`. If you wish to process a single input binary mask with pyRootHair, you must specify `-p single`. 

### Random Forest Pipeline
If you do not have access to a GPU, it is possible to train your own random forest segmentation model. The random forest segmentation model is nowhere near as powerful as the CNN, and as such, will struggle with image anomalies or noise. Please note this is an experimental feature, and should be used with caution.

You will need to ensure that all the images are relatively consistent in terms of lighting, appearance, root hair morphology, and have the same input dimensions. Should your images vary for these traits, you will need to train separate random forest models for different batches of images.

#### Training the Random Forest Model
To train a random forest model, you will need to train the model on a single representative example of an image, and a corresponding binary mask of the image. See [this](https://github.com/iantsang779/pyRootHair?tab=readme-ov-file#generating-binary-masks) section for details on how to generate suitable binary masks.

Once you have generated a suitable mask, you can train a random forest model like so:

`pyroothair_train_rf_model --train_img /path/to/representative/training/image/example --train_mask /path/to/generated/binary/mask --model_output /path/to/output/rf_model/`

If successful, you should see `...RFC model saved as /path/to/output/rf_model.joblib...`, indicating the random forest model has been saved in `--model_output`. There are some additional flags/arguments that allow you to toggle the behaviour of how the random forest model is trained, please see the [documentation] (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from scikit-learn on the `RandomForestClassifier` for more information. 

**`--sigma_min`**: Minimum sigma (blurring factor) for feature extraction from the input image. Default = 1
**`--sigma_max`**: Maximum sigma (blurring factor) for feature extraction from input image. Default = 4
**`--n_estimators`**: Number of trees in the Random Forest Classifier. Default = 50
**`--max_depth`**: Maximum depth of the Random Forest Classifier. Default = 10
**`--max_samples`**: Number of samples extracted from image features to train each estimator. Default = 0.05

#### Deploying the Random Forest Model
Once your random forest model is trained, you can deploy it like so:

`pyroothair -i /path/to/input/image/folder -b batch_id -o /path/to/output/folder -p random_forest -rfc_model_path /path/to/rf_model.joblib` 

The command is the same as before, but you specify to run the random forest pipeline with `-p random_forest`, and provide the path to the trained model for `--rfc_model_path`.

### Single Mask Pipeline
If you wish, you can also run pyRootHair on a single, user generated binary mask of an input image. See [this] (https://github.com/iantsang779/pyRootHair?tab=readme-ov-file#generating-binary-masks) for instructions on generating binary masks. 

To run pyRootHair on a single binary mask (with classes converted!):

`pyroothair --input_mask /path/to/converted/binary/mask -p single -o /path/to/output`

Note that you no longer require `-i` or `-b` when using this pipeline option.

## Generating Binary Masks
pyRootHair will accept binary masks of any images as long as they are arrays of 0s, 1s and 2s. It is recommended that you use the ilastik software (https://www.ilastik.org/) to generate the masks, as it is simple and requires minimal expertise to use.

This section is not a tutorial on how to use ilastik, rather, a demonstration on what the masks need to look like if you wish to generate your own masks suitable for pyroothair.

1.) Under the `1. Input Data` tab, upload your raw image(s). Ensure that the input image only has 3 channels! 
2.) Select all features under the `Feature Selection` tab.  
3.) Specify the following label categories. The label order **must** be in the exact order as shown here!  

![alt text](demo/ilastik_classes.png)

4.) Ensure that the root hairs wrap around the root, especially at the root tip:

![alt text](demo/demo_mask.png)

5.) After generating the mask, select `Source: Simple Segmentation` under `4. Prediction Export`. Click on `Choose Export Image Settings`, and set the output file format to `.png`, then hit `Export All`. 

6.) Once generated, the mask should be converted such that each pixel is a 0, 1 or 2 in the array. By default, ilastik saves each pixel associated with the background as 1, root hair as 2, and root as 3. 

If you are using the generated mask to train a random forest model, ***IGNORE the rest of this step!***. However, if you are going to load the mask into pyRootHair with `-p 3` and `--input_mask`, please read on:

You will need to convert the pixel classes of the generated binary mask as follows: 

`pyroothair_convert_mask -i path/to/your/generated/mask`

You should see a the following message if the conversion has been successful: `...Saved converted input mask XXX in ZZZ...`. You can now process this mask with pyRootHair: `pyroothair -p 3 --input_mask /path/to/converted/mask`.

## pyRootHair Workflow

If you are interested in learning how pyRootHair works behind the scenes, please check out [this] (https://github.com/iantsang779/pyRootHair/blob/main/workflow.md) in-depth walk through.



















