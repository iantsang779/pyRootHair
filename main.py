import argparse
import time
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os
import sys

from cnn import nnUNet
from images import ImageLoader
from params import GetParams
from pathlib import Path
from random_forest import ForestTrainer
from root import Root
from skeleton import Skeleton

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair',
                                     description='pyRootHair Arguments')
    
    parser.add_argument('--no_gpu', help='Toggle whether to run pyRootHair with, or without a GPU. Only specify this flag if a GPU is unavailable.', dest='no_gpu', action='store_true')
    parser.add_argument('--input', help='Filepath to directory containing input image(s).', type=str, nargs='?', dest='img_dir')
    parser.add_argument('--run_id', help='Unique ID for each batch of input images', type=str, nargs='?', dest='run_id')
    parser.add_argument('--model_path', help='Filepath to nnU-Net segmentation model', type=str, dest='model_path')
    parser.add_argument('--override_model_path', help='Filepath to custom nnU-Net segmentation model', type=str, dest='custom_model_path')
    parser.add_argument('--override_model_dataset', help='Dataset ID for nnUNetv2. Required if specifying --override_model_path', type=str, dest='custom_dataset_id', required='--override_model_path' in sys.argv)
    parser.add_argument('--override_model_planner', help='Model plans for nnUNetv2. Required if specifying --override_model_path', type=str, dest='custom_model_planner', required='--override_model_path' in sys.argv)
    parser.add_argument('--rfc_model_path', help='Filepath to trained Random Forest Classifier model. Required if specifying --no_gpu', type=str, dest='rfc_model_path')
    parser.add_argument('--resolution', help='Bin size (pixels) for measurements along each root hair segment. Smaller bin sizes yield more data points per root (default = 20 px)', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--rhd_filt', help='Area threshold to remove small areas from area list; sets area for a particular bin to 0 when below the value (default = 180 px^2)', type=int, nargs='?', dest='area_filt', default=180)
    parser.add_argument('--rhl_filt', help='Length threshold to remove small lengths from length list; sets length for a particular bin to 0 when below the value (default = 14px)', type=int, nargs='?', dest='length_filt', default=14)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images (default = 127.5 px)', type=int, nargs='?', dest='conv', default=127.5)
    parser.add_argument('--output', help='Filepath to save data.', type=str, dest='save_path')
    parser.add_argument('--plot_segmentation', help='Save model segmentation results in --output directory. Useful for debugging.', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot_transformation', help='Save diagnostic plot showing root straightening in --output directory. Useful for debugging.', dest='show_transformation', action='store_true')
    parser.add_argument('--plot_summary', help='Save summary plots for each input image in --output directory.', dest='show_summary', action='store_true')

    return parser.parse_args(), parser

def check_arguments_gpu():
    args, parser = parse_args()

    missing_args = []
        # check necessary arguments are supplied
    if args.img_dir is None:
        missing_args.append('--input')
    if args.run_id is None:
        missing_args.append('--run_id')
    if args.model_path is None:
        missing_args.append('--model_path')
    if missing_args:
        parser.error(f'The following arguments are required unless --no_gpu is specified: {missing_args}')

def check_arguments_nogpu():
    args, parser = parse_args()

    if args.no_gpu:
        if args.rfc_model_path is None:
            parser.error(f'The following argument is required when specifying --no_gpu: --rfc_model_path.')
        if args.model_path or args.custom_model_path:
            parser.error(f'Invalid argument with --no_gpu! --no_gpu should only be run with --rfc_model_path and --output.')

def main():
    args, _ = parse_args()

    model = nnUNet()
    model.check_gpu() # determine which model to load depending on GPU availability
    
    start = time.perf_counter()

    summary = pd.DataFrame()
    raw = pd.DataFrame()

    if model.gpu_exists and not args.no_gpu: # check if GPU exists
        
        check_arguments_gpu()

        for img in os.listdir(args.img_dir): # loop through all input images, modify and save with nnUNet prefix
            im_loader = ImageLoader()
            im_loader.read_images(args.img_dir, img)
            im_loader.resize_height()
            im_loader.resize_width()
            im_loader.resize_channel()
            im_loader.setup_dir(args.img_dir, args.run_id)
            im_loader.save_resized_image()
            model.setup_nnunet_paths() # set up nnUNet results path
        
        if args.custom_model_path is not None: # check whether path to own nnUNet model has been specified
            model.load_model(args.custom_model_path)    
        else: # load and run inference with pre-trained model if --custom_model_path is not supplied
            model.load_model(args.model_path) 
        
        model.run_inference(args.run_id, args.custom_dataset_id, args.custom_model_planner) # generate predicted masks

        mask_path = Path(args.img_dir).parent/'masks'/args.run_id

        for mask_file in os.listdir(mask_path): # loop through each predicted mask
            print(f'\n...Processing {mask_file}...')
            init_mask = iio.imread(os.path.join(mask_path, mask_file))
            root_mask = (init_mask == 2)

            skeleton = Skeleton() 
            clean_root = skeleton.extract_root(root_mask)
            sk_y, sk_x = skeleton.skeletonize(clean_root)
            sk_spline, sk_height = skeleton.skeleton_params(sk_x, sk_y)
            med_x, med_y = skeleton.calc_skeleton_midline(sk_spline, sk_height)
            rotated_mask = skeleton.calc_rotation(med_x, med_y, init_mask) 

            rotated_root_mask = (rotated_mask == 2) 

            clean_root_rotated = skeleton.extract_root(rotated_root_mask)
            sk_r_y, sk_r_x = skeleton.skeletonize(clean_root_rotated)
            sk_r_spline, sk_r_height = skeleton.skeleton_params(sk_r_x, sk_r_y)
            med_r_x, med_r_y = skeleton.calc_skeleton_midline(sk_r_spline, sk_r_height)
            skeleton.add_endpoints(med_r_x, med_r_y)
            skeleton.calc_skel_euclidean()
            skeleton.generate_buffer_coords(rotated_mask)
            straight_mask = skeleton.straighten_image(rotated_mask)

            rt = Root(straight_mask)
            final_root = rt.check_root_tip()
            rt.find_root_tip()
            root_hairs = rt.trim_rh_mask()
            rt.split_root_coords(root_hairs)

            data = GetParams(root_hairs)
            data.sliding_window(args.height_bin_size)
            data.clean_data(args.area_filt, args.length_filt)
            data.calibrate_data(args.conv)
            data.calculate_avg_root_thickness(final_root, args.conv)
            data.calculate_uniformity()
            data.calculate_growth()
            # datetime = data.get_metadata(image.image_metadata)
        
            summary_df, raw_df = data.generate_table(mask_file.split('.')[0])

            summary = pd.concat([summary_df,summary])
            raw = pd.concat([raw_df, raw])

            if args.show_transformation:
                skeleton.visualize_transformation(init_mask, args.save_path, mask_file.split('.')[0]) 

            if args.show_segmentation:
                plt.imsave(os.path.join(args.save_path,f'{mask_file.split('.')[0]}_mask.png'), straight_mask)
            
            if args.show_summary:
                data.plot_summary(mask_file.split('.')[0])

        print(f'\n{summary}')
        print(f'\n{raw}')

        if args.save_path:
            summary.to_csv(f'{args.save_path}_summary.csv')
            raw.to_csv(f'{args.save_path}_raw.csv')
        
        print(f'\nTotal runtime for current batch of images: {time.perf_counter()-start:.2f} seconds.')

    elif args.no_gpu: # if GPU is not available
        check_arguments_nogpu()
        rf = ForestTrainer()
        model = rf.load_model(args.rfc_model_path)

    else:
        raise ValueError('Missing a GPU. Please add the --no_gpu flag, and specify the path to the trained random forest model with --rfc_model_path')


if __name__ == '__main__':
    main()
