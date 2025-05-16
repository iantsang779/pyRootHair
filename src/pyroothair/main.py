import argparse
import time
import pandas as pd
import imageio.v3 as iio
import os
import torch

from pathlib import Path
from pyroothair.cnn import nnUNetv2
from pyroothair.images import ImageLoader
from pyroothair.random_forest import ForestTrainer
from pyroothair.pipeline import CheckArgs, Pipeline

description = '''
Thank you for using pyRootHair!
-------------------------------
Please read the tutorial documentation on the github repository: https://github.com/iantsang779/pyRootHair

Basic usage: pyroothair -i /path/to/image/folder -b unique_ID_for_folder -o /path/to/output/folder

Please cite the following paper when using pyRootHair: xxxxxx

Author: Ian Tsang
Contact: ian.tsang@niab.com
''' 

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    ### Required Arguments    
    parser.add_argument('-i', '--input', help='Filepath to directory containing input image(s).', type=str, nargs='?', dest='img_dir')
    parser.add_argument('-b', '--batch_id', help='Unique ID for each batch of input images', type=str, nargs='?', dest='batch_id')
    parser.add_argument('-o','--output', help='Filepath to save data. Must be a different directory relative to the input image directory.', type=str, dest='save_path')

    ### Optional Arguments
    # pyRootHair Arguments to Toggle Output
    parser.add_argument('-p', '--pipeline', 
                        help="""Pipeline ID to determine which pipeline to run. 

                        cnn - Perform segmentation using pre-trained nnUNet model. Can be run with or without GPU. This is the default option. 
                        random_forest - Perform segmentation using a trained random forest classifier. Does not require a GPU. 
                        single - Directly extract traits from a user generated binary mask. No inference is run.""", 
                        default='cnn', choices=['cnn','random_forest','single'], type=str, nargs='?', dest='pipeline_choice')
    parser.add_argument('--resolution', help='Bin size defining measurement intervals along each root hair segment. Default = 20 px', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images. Default = 102 px', nargs='?', type=int, dest='conv', default=102)
    parser.add_argument('--frac', help='Degree of smoothing of lowess regression line to model average root hair length per input image. Value must be between 0 and 1. See statsmodels.nonparametric.smoothers_lowess.lowess for more details. Default = 0.1', type=float, nargs='?', dest='frac', default=0.1)
    parser.add_argument('--plot_segmentation', help='toggle plotting of predicted binary masks for each image (straightened mask, root hair segments, and cropped root hair segments). Must provide a valid filepath for --output', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot_transformation', help='toggle plotting of co-ordinates illustrating how each input image is warped and straightened. Useful for debugging any strangely warped masks. Must provide a valid filepath for --output', dest='show_transformation', action='store_true')
    parser.add_argument('--plot_summary', help='toggle plotting of summary plots describing RHL and RHD for each image. Must provide a valid filepath for --output', dest='show_summary', action='store_true')

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
    if args.pipeline_choice == 'cnn': 

        model = nnUNetv2(args.img_dir, args.batch_id)
        model.check_gpu() # determine which model to load depending on GPU availability
        check_args.check_arguments_gpu()
        if not Path(args.save_path).exists(): # check if --output path exists, create it if not
            Path(args.save_path).mkdir(parents=True, exist_ok=True)
        model.download_model()
        
        device = torch.device('cuda', 0) if model.gpu_exists else torch.device('cpu')
        
        for img in os.listdir(args.img_dir): # loop through all input images, modify and save with nnUNet prefix
            im_loader = ImageLoader()
            im_loader.read_images(args.img_dir, img)
            im_loader.resize_image()
            im_loader.resize_channel()
            im_loader.setup_dir(args.save_path, args.batch_id)
            im_loader.save_resized_image()

        model.initialize_model(device)
        model.run_inference(args.save_path)

        mask_path = Path(args.save_path)/'masks'/args.batch_id

        failed_images = []

        for mask_file in os.listdir(mask_path): # loop through each predicted mask
            if mask_file.endswith('.png'):
                main = Pipeline(check_args)
                init_mask = iio.imread(os.path.join(mask_path, mask_file))
                print(f'\n...Processing {mask_file}...')
                s, r = main.run_pipeline(init_mask, mask_file) # run pipeline for each image
                if isinstance(s, pd.DataFrame):
                    summary = pd.concat([s,summary]) # add data from each image to the correct data frame
                    raw = pd.concat([r,raw])
                else:  
                    failed_images.append(mask_file)
        
        print(f'\n{summary}')
        print(f'\n{raw}')

        data_path = Path(args.save_path) / 'data' / args.batch_id
        data_path.mkdir(exist_ok=True, parents=True)

        summary.to_csv(os.path.join(data_path, f'{args.batch_id}_summary.csv'))
        raw.to_csv(os.path.join(data_path, f'{args.batch_id}_raw.csv'))
        
        if failed_images:
            print(f'\nAn error occurred with the following image(s) and were skipped: {failed_images}')

        print(f'\nTotal runtime for batch_id {args.batch_id}: {time.perf_counter()-start:.2f} seconds.')

    #! 2 - Random Forest Pipeline
    elif args.pipeline_choice == 'random_forest': 
        check_args.check_arguments_rfc()
        rf = ForestTrainer()
        model = rf.load_model(args.rfc_model_path) # load trained random forest model
        if not Path(args.save_path).exists():
                Path(args.save_path).mkdir(parents=True, exist_ok=True)
        failed_images = []

        for img in os.listdir(args.img_dir): 
            if img.endswith('.png'):
                mask = rf.predict(args.img_dir, img, args.sigma_min, args.sigma_max, model)
                init_mask = rf.reconvert_mask_class(mask) # check mask classes are 0, 1, 2
                main = Pipeline(check_args)
                s, r = main.run_pipeline(init_mask, img)
                if isinstance(s, pd.DataFrame):
                    summary = pd.concat([s,summary]) # add data from each image to the correct data frame
                    raw = pd.concat([r,raw])
                else:  
                    failed_images.append(init_mask)

        print(f'\n{summary}')
        print(f'\n{raw}')

        data_path = Path(args.save_path) / 'data' / args.batch_id
        data_path.mkdir(exist_ok=True, parents=True)

        summary.to_csv(os.path.join(data_path, f'{args.batch_id}_summary.csv'))
        raw.to_csv(os.path.join(data_path, f'{args.batch_id}_raw.csv'))

        if failed_images:
            print(f'\nAn error occurred with the following image(s), and have been skipped: {failed_images}')

        print(f'\nTotal runtime for batch_id {args.batch_id}: {time.perf_counter()-start:.2f} seconds.')

    # ! 3 - Single Mask Pipeline
    else:
        check_args.check_arguments_single_mask()
        
        if not Path(args.save_path).exists():
            Path(args.save_path).mkdir(parents=True, exist_ok=True)

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

        data_path = Path(args.save_path) / 'data' / args.batch_id
        data_path.mkdir(exist_ok=True, parents=True)
        
        summary.to_csv(os.path.join(data_path, f'{fname}_summary.csv'))
        raw.to_csv(os.path.join(data_path, f'{fname}_raw.csv'))

        print(f'\nTotal runtime for image {fname}: {time.perf_counter()-start:.2f} seconds.')
        
        
if __name__ == '__main__':
    main()
