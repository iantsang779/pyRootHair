import argparse
import time
import pandas as pd
import imageio.v3 as iio
import os
import torch

from cnn import nnUNetv2
from images import ImageLoader
from pathlib import Path
from random_forest import ForestTrainer
from pipeline import CheckArgs, Pipeline

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair',
                                     description='pyRootHair Arguments')

    ### Required Arguments    
    parser.add_argument('-i', '--input', help='Filepath to directory containing input image(s).', type=str, nargs='?', dest='img_dir')
    parser.add_argument('-b', '--batch_id', help='Unique ID for each batch of input images', type=str, nargs='?', dest='batch_id')

    ### Optional Arguments
    # pyRootHair Arguments to Toggle Output
    parser.add_argument('-p', '--pipeline', 
                        help="""Pipeline ID to determine which pipeline to run. 

                        1 - Perform segmentation using pre-trained nnUNet model. Can be run with or without GPU. This is the default option. 
                        2 - Perform segmentation using a trained random forest classifier. Does not require a GPU. 
                        3 - Directly extract traits from a user generated binary mask. No segmentation is performed.""", 
                        default=1, choices=[1,2,3], type=int, nargs='?', dest='pipeline_choice')
    parser.add_argument('--resolution', help='Bin size defining measurement intervals along each root hair segment. Default = 20 px', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--split_segments', help='Padding around root tip/end of root in image (in pixels) to split root hair fragments. Default = 20 px.', type=int, nargs='?', dest='padding', default=20)
    parser.add_argument('--rhd_filt', help='Area threshold to remove small areas from area list; sets area for a particular bin to 0 when below the value. Default = 0.03 mm^2)', type=float, nargs='?', dest='area_filt', default=0.03)
    parser.add_argument('--rhl_filt', help='Length threshold to remove small lengths from length list; sets length for a particular bin to 0 when below the value. Default = 0.2 mm', type=float, nargs='?', dest='length_filt', default=0.2)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images. Default = 102 px', nargs='?', type=int, dest='conv', default=102)
    parser.add_argument('--frac', help='Degree of smoothing of lowess regression line to model average root hair length per input image. Value must be between 0 and 1. See statsmodels.nonparametric.smoothers_lowess.lowess for more details. Default = 0.15', type=float, nargs='?', dest='frac', default=0.15)
    parser.add_argument('-o','--output', help='Filepath to save data. Must be a different directory relative to the input image directory.', type=str, dest='save_path')
    parser.add_argument('--plot_segmentation', help='Save model segmentation results in --output directory. Useful for debugging.', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot_transformation', help='Save diagnostic plot showing root straightening in --output directory. Useful for debugging.', dest='show_transformation', action='store_true')
    parser.add_argument('--plot_summary', help='Save summary plots for each input image in --output directory.', dest='show_summary', action='store_true')

    # Custom nnUNet Model Arguments
    parser.add_argument('--override_model_path', help='Filepath to custom nnUNet model.', type=str, dest='override_model_path')
    parser.add_argument('--override_model_checkpoint', help='Checkpoint file for custom nnUNet model', type=str, dest='override_model_chkpoint')
    
    # Random Forest Pipeline Arguments
    parser.add_argument('--rfc_model_path', help='Filepath to trained Random Forest Classifier model.', type=str, dest='rfc_model_path')
    parser.add_argument('--sigma_min', help='Minimum sigma for feature extraction. Required if specifying --rfc_model_path. Default = 1', dest='sigma_min', type=int, default=1)
    parser.add_argument('--sigma_max', help='Maximum sigma for feature extraction. Required if specifying --rfc_model_path. Default = 4', dest='sigma_max', type=int, default=4)

    # Single Mask Pipeline Arguments
    parser.add_argument('--input_mask', help='Filepath to a single mask.', type=str, nargs='?', dest='input_mask')

    return parser.parse_args(), parser
#

def main():
    args, parser = parse_args()
    check_args = CheckArgs(args, parser)

    raw = pd.DataFrame() # initialize empty data frames to append to in run_pipeline()
    summary = pd.DataFrame()

    start = time.perf_counter()

    #! 1 - Main pipeline using nnUNet
    if args.pipeline_choice == 1: 

        model = nnUNetv2(args.img_dir, args.batch_id)
        model.check_gpu() # determine which model to load depending on GPU availability
        check_args.check_arguments_gpu()

        if model.gpu_exists: # set device to available GPU or CPU
            device = torch.device('cuda',0) 
        else:
            device = 'cpu'
        
        for img in os.listdir(args.img_dir): # loop through all input images, modify and save with nnUNet prefix
            im_loader = ImageLoader()
            im_loader.read_images(args.img_dir, img)
            im_loader.resize_image()
            im_loader.resize_channel()
            im_loader.setup_dir(args.img_dir, args.batch_id)
            im_loader.save_resized_image()

        if args.override_model_path is None: # if using default model

            model.initialize_model(device)
            model.run_inference()
        
        else: # if a custom model path is loaded
            model.initialize_model(device, args.override_model_path, args.override_model_checkpoint)
            model.run_inference()

        mask_path = Path(args.img_dir).parent/'masks'/args.batch_id

        for mask_file in os.listdir(mask_path): # loop through each predicted mask
            if mask_file.endswith('.png'):
                main = Pipeline(check_args)
                init_mask = iio.imread(os.path.join(mask_path, mask_file))
                print(f'\n...Processing {mask_file}...')
                s, r = main.run_pipeline(init_mask, mask_file) # run pipeline for each image
                
                summary = pd.concat([s,summary]) # add data from each image to the correct data frame
                raw = pd.concat([r,raw])
        
        print(f'\n{summary}')
        print(f'\n{raw}')

        if args.save_path:
            summary.to_csv(os.path.join(args.save_path, f'{args.batch_id}_summary.csv'))
            raw.to_csv(os.path.join(args.save_path, f'{args.batch_id}_raw.csv'))

        print(f'\nTotal runtime for batch_id {args.batch_id}: {time.perf_counter()-start:.2f} seconds.')

    #! 2 - Random Forest Pipeline
    elif args.pipeline_choice == 2: 
        check_args.check_arguments_rfc()
        rf = ForestTrainer()
        model = rf.load_model(args.rfc_model_path) # load trained random forest model
    
        for img in os.listdir(args.img_dir): 
            if img.endswith('.png'):
                mask = rf.predict(args.img_dir, img, args.sigma_min, args.sigma_max, model)
                init_mask = rf.reconvert_mask_class(mask) # check mask classes are 0, 1, 2
                main = Pipeline(check_args)
                s, r = main.run_pipeline(init_mask, img)

                summary = pd.concat([s,summary]) # add data from each image to the correct data frame
                raw = pd.concat([r,raw])

        print(f'\n{summary}')
        print(f'\n{raw}')

        if args.save_path:
            summary.to_csv(f'{args.batch_id}_summary.csv')
            raw.to_csv(f'{args.batch_id}_raw.csv')

        print(f'\nTotal runtime for batch_id {args.batch_id}: {time.perf_counter()-start:.2f} seconds.')

    # ! 3 - Single Mask Pipeline
    else:
        check_args.check_arguments_single_mask()
        
        if '/' in args.input_mask: # get image name
            fname = args.input_mask.split('/')[-1].split('.')[0]
        else:
            fname = args.input_mask.split('.')[0]

        mask = iio.imread(args.input_mask)
        mask = check_args.convert_mask(mask)
        main = Pipeline(check_args)
        s, r = main.run_pipeline(mask, fname)
        
        summary = pd.DataFrame(s)
        raw = pd.DataFrame(r)

        print(f'\n{summary}')
        print(f'\n{raw}')

        if args.save_path:
            summary.to_csv(Path(f'{args.save_path}/{fname}_summary.csv'))
            raw.to_csv(Path(f'{args.save_path}/{fname}_raw.csv'))

        print(f'\nTotal runtime for image {fname}: {time.perf_counter()-start:.2f} seconds.')
        
        
if __name__ == '__main__':
    main()
