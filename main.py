import argparse
import time
import pandas as pd
import imageio.v3 as iio
import os
import sys

from cnn import nnUNet
from images import ImageLoader
from pathlib import Path
from random_forest import ForestTrainer
from pipeline import CheckArgs, Pipeline

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair',
                                     description='pyRootHair Arguments')
    
    parser.add_argument('--input', help='Filepath to directory containing input image(s).', type=str, nargs='?', dest='img_dir')
    parser.add_argument('--batch_id', help='Unique ID for each batch of input images', type=str, nargs='?', dest='batch_id')
    parser.add_argument('--model_path', help='Filepath to nnU-Net segmentation model', type=str, dest='model_path')
    parser.add_argument('--override_model_path', help='Filepath to custom nnU-Net segmentation model', type=str, dest='custom_model_path')
    parser.add_argument('--override_model_dataset', help='Dataset ID for nnUNetv2. Required if specifying --override_model_path', type=str, dest='custom_dataset_id', required='--override_model_path' in sys.argv)
    parser.add_argument('--override_model_planner', help='Model plans for nnUNetv2. Required if specifying --override_model_path', type=str, dest='custom_model_planner', required='--override_model_path' in sys.argv)
    parser.add_argument('--rfc_model_path', help='Filepath to trained Random Forest Classifier model. Required if specifying --no_gpu', type=str, dest='rfc_model_path')
    parser.add_argument('--input_mask', help='Filepath to a single mask.', type=str, nargs='?', dest='input_mask')
    parser.add_argument('--sigma_min', help='Minimum sigma for feature extraction. Required if specifying --no_gpu and --rfc_model_path. Default = 1', dest='sigma_min', type=int, default=1)
    parser.add_argument('--sigma_max', help='Maximum sigma for feature extraction. Required if specifying --no_gpu and --rfc_model_path. Default = 4', dest='sigma_max', type=int, default=4)
    parser.add_argument('--resolution', help='Bin size defining measurement intervals along each root hair segment. Default = 20 px', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--rhd_filt', help='Area threshold to remove small areas from area list; sets area for a particular bin to 0 when below the value. Default = 180 px^2)', type=int, nargs='?', dest='area_filt', default=180)
    parser.add_argument('--rhl_filt', help='Length threshold to remove small lengths from length list; sets length for a particular bin to 0 when below the value. Default = 14 px', type=int, nargs='?', dest='length_filt', default=14)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images. Default = 127.5 px', type=int, nargs='?', dest='conv', default=127.5)
    parser.add_argument('--output', help='Filepath to save data. Must be a different directory relative to the input image directory.', type=str, dest='save_path')
    parser.add_argument('--plot_segmentation', help='Save model segmentation results in --output directory. Useful for debugging.', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot_transformation', help='Save diagnostic plot showing root straightening in --output directory. Useful for debugging.', dest='show_transformation', action='store_true')
    parser.add_argument('--plot_summary', help='Save summary plots for each input image in --output directory.', dest='show_summary', action='store_true')

    return parser.parse_args(), parser
#

def main():
    args, parser = parse_args()
    check_args = CheckArgs(args, parser)

    model = nnUNet(args.img_dir, args.batch_id)
    model.check_gpu() # determine which model to load depending on GPU availability
    
    raw = pd.DataFrame() # initialize empty data frames to append to in run_pipeline()
    summary = pd.DataFrame()

    start = time.perf_counter()

    if model.gpu_exists: #! 1. Main Pipeline (With GPU)

        check_args.check_arguments_gpu()

        if args.custom_model_path is None: # if using default model
        
            for img in os.listdir(args.img_dir): # loop through all input images, modify and save with nnUNet prefix
                im_loader = ImageLoader()
                im_loader.read_images(args.img_dir, img)
                im_loader.resize_height()
                im_loader.resize_width()
                im_loader.resize_channel()
                im_loader.setup_dir(args.img_dir, args.batch_id)
                im_loader.save_resized_image()
                model.setup_nnunet_paths() # set up nnUNet results path
        
            model.load_model(args.model_path) 
            model.run_inference() # generate predicted masks

        else: # if a custom model path is loaded
            model.load_model(args.custom_model_path)    
            model.run_inference(args.custom_dataset_id, args.custom_model_planner) # generate predicted masks            

        mask_path = Path(args.img_dir).parent/'masks'/args.batch_id

        for mask_file in os.listdir(mask_path): # loop through each predicted mask
            if mask_file.endswith('.png'):
                main = Pipeline(check_args)
                init_mask = iio.imread(os.path.join(mask_path, mask_file))
                s, r = main.run_pipeline(init_mask, mask_file) # run pipeline for each image
                
                summary = pd.concat([s,summary]) # add data from each image to the correct data frame
                raw = pd.concat([r,raw])
        
        print(f'\n{summary}')
        print(f'\n{raw}')

        if args.save_path:
            summary.to_csv(f'{args.batch_id}_summary.csv')
            raw.to_csv(f'{args.batch_id}_raw.csv')

        print(f'\nTotal runtime for batch_id {args.batch_id}: {time.perf_counter()-start:.2f} seconds.')


    elif args.input_mask: #! 2. Secondary pipeline, process single binary masks at a time
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

    else: #! 3. Secondary pipeline, use a trained Random Forest Classifier to process a batch of images
        
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
if __name__ == '__main__':
    main()
