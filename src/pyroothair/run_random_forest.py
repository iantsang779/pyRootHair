import argparse
import time
import pandas as pd
import os
import imageio.v3 as iio

from pathlib import Path
from pyroothair.train_random_forest import ForestTrainer
from pyroothair.pipeline import CheckArgs, Pipeline

description = '''
Thank you for using pyRootHair!
-------------------------------

Run pyRootHair using a trained random forest model.

Please read the tutorial documentation on the github repository: https://github.com/iantsang779/pyRootHair

Basic usage: pyroothair_run_random_forest -i /path/to/image/folder -b unique_ID_for_folder -o /path/to/output/folder --rfc_model_path /path/to/trained/rfc/model

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
    parser.add_argument('-b', '--batch-id', help='Unique ID for each batch of input images', type=str, nargs='?', dest='batch_id')
    parser.add_argument('-o','--output', help='Filepath to save data. Must be a different directory relative to the input image directory.', type=str, dest='save_path')

    parser.add_argument('--resolution', help='Bin size defining measurement intervals along each root hair segment. Default = 20 px', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images. Default = 102 px', nargs='?', type=int, dest='conv', default=102)
    parser.add_argument('--frac', help='Degree of smoothing of lowess regression line to model average root hair length per input image. Value must be between 0 and 1. See statsmodels.nonparametric.smoothers_lowess.lowess for more details. Default = 0.1', type=float, nargs='?', dest='frac', default=0.1)
    parser.add_argument('--length-cutoff', help='Set maximum length (mm) of root (as distance from the root tip) to standardize measurements for all images in the input batch. Please only use this argument once you have run pyroothair once on the existing batch.', dest='length_cutoff', type=float, default=None)
    parser.add_argument('--plot-segmentation', help='Toggle plotting of predicted binary masks for each image (straightened mask and root hair segments). Must provide a valid filepath for --output', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot-transformation', help='Toggle plotting of co-ordinates illustrating how each input image is warped and straightened. Useful for debugging any strangely warped masks. Must provide a valid filepath for --output', dest='show_transformation', action='store_true')
    parser.add_argument('--plot-summary', help='Toggle plotting of summary plots describing RHL and RHD for each image. Must provide a valid filepath for --output', dest='show_summary', action='store_true')

    # Random Forest Pipeline Arguments
    parser.add_argument('-rfc', '--rfc-model-path', help='Filepath to trained Random Forest Classifier model.', type=str, dest='rfc_model_path')
    parser.add_argument('--sigma-min', help='Minimum sigma for feature extraction. Required if specifying --rfc_model_path. Default = 1', dest='sigma_min', type=int, default=1)
    parser.add_argument('--sigma-max', help='Maximum sigma for feature extraction. Required if specifying --rfc_model_path. Default = 4', dest='sigma_max', type=int, default=4)

    return parser.parse_args(), parser

def main():
    args, parser = parse_args()
    check_args = CheckArgs(args, parser)

    raw = pd.DataFrame() # initialize empty data frames to append to in run_pipeline()
    summary = pd.DataFrame()    
    start = time.perf_counter()

    check_args.check_arguments_rfc()
    rf = ForestTrainer()
    model = rf.load_model(args.rfc_model_path) # load trained random forest model

    if not Path(args.save_path).exists():
            Path(args.save_path).mkdir(parents=True, exist_ok=True)
    failed_images = []

    input_directory = Path(args.img_dir)
    input_images = sorted([i for i in os.listdir(input_directory) if i.endswith('.png')])

    for img in input_images: 
        img_file = iio.imread(os.path.join(input_directory, img))
        if img_file.shape[2] == 4:
             img_file = img_file[:,:,:3]
        mask = rf.predict(args.img_dir, img, args.sigma_min, args.sigma_max, model)
        init_mask = rf.reconvert_mask_class(mask) # check mask classes are 0, 1, 2
        main = Pipeline(args, parser, args.img_dir, img)
        s, r = main.run_pipeline(init_mask, img)
        if len(s) > 0:
            summary = pd.concat([s,summary]) # add data from each image to the correct data frame
            raw = pd.concat([r,raw])
        else:  
            failed_images.append(img)

    print(f'\n{summary}')
    print(f'\n{raw}')

    data_path = Path(args.save_path) / 'data' / args.batch_id
    data_path.mkdir(exist_ok=True, parents=True)

    summary.to_csv(os.path.join(data_path, f'{args.batch_id}_summary.csv'))
    raw.to_csv(os.path.join(data_path, f'{args.batch_id}_raw.csv'))

    if failed_images:
        print(f'\nAn error occurred with the following image(s), and have been skipped: {failed_images}')

    print(f'\nTotal runtime for batch_id {args.batch_id}: {time.perf_counter()-start:.2f} seconds.')

if __name__ == '__main__':
     main()
     