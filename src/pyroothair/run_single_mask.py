import argparse
import time
import pandas as pd
import imageio.v3 as iio
import os

from pathlib import Path
from pyroothair.pipeline import CheckArgs, Pipeline

description = '''
Thank you for using pyRootHair!
-------------------------------

Run pyroothair on a single (user generated) binary mask.

Please read the tutorial documentation on the github repository: https://github.com/iantsang779/pyRootHair

Basic usage: pyroothair_run_single_mask -m /path/to/binary/mask -i /path/to/image/of/binary/mask -b unqiue_id_for_mask/image -o /path/to/output/folder

Please cite the following paper when using pyRootHair: xxxxxx

Author: Ian Tsang
Contact: ian.tsang@niab.com
''' 

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--mask', help='Filepath to a single mask.', type=str, nargs='?', dest='input_mask')
    parser.add_argument('-i', '--image', help='Filepath to image corresponding to the mask', type=str, nargs='?', dest='input_image')
    parser.add_argument('-b', '--batch-id', help='Unique ID for each batch of input images', type=str, nargs='?', dest='batch_id')
    parser.add_argument('-o','--output', help='Filepath to save data. Must be a different directory relative to the input image directory.', type=str, dest='save_path')

    parser.add_argument('--resolution', help='Bin size defining measurement intervals along each root hair segment. Default = 20 px', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images. Default = 102 px', nargs='?', type=int, dest='conv', default=102)
    parser.add_argument('--frac', help='Degree of smoothing of lowess regression line to model average root hair length per input image. Value must be between 0 and 1. See statsmodels.nonparametric.smoothers_lowess.lowess for more details. Default = 0.1', type=float, nargs='?', dest='frac', default=0.1)
    parser.add_argument('--length-cutoff', help='Set maximum length (mm) of root (as distance from the root tip) to standardize measurements for all images in the input batch. Please only use this argument once you have run pyroothair once on the existing batch.', dest='length_cutoff', type=float, default=None)
    parser.add_argument('--plot-transformation', help='Toggle plotting of co-ordinates illustrating how each input image is warped and straightened. Useful for debugging any strangely warped masks. Must provide a valid filepath for --output', dest='show_transformation', action='store_true')
    parser.add_argument('--plot-segmentation', help='toggle plotting of predicted binary masks for each image (straightened mask, root hair segments, and cropped root hair segments). Must provide a valid filepath for --output', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot-summary', help='Toggle plotting of summary plots describing RHL and RHD for each image. Must provide a valid filepath for --output', dest='show_summary', action='store_true')

    return parser.parse_args(), parser

def main():
    args, parser = parse_args()
    check_args = CheckArgs(args, parser)

    raw = pd.DataFrame() # initialize empty data frames to append to in run_pipeline()
    summary = pd.DataFrame()    
    start = time.perf_counter()

    check_args.check_arguments_single_mask()
        
    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if '/' in args.input_mask: # get image name
        fname = args.input_mask.split('/')[-1].split('.')[0]
    else:
        fname = args.input_mask.split('.')[0]

    mask = iio.imread(Path(args.input_mask))
    mask = check_args.convert_mask(mask)
    img_dir = str(Path(args.input_image).parent)
    img_path = str(Path(args.input_image))

    if '/' in img_path:
        img_path = img_path.split('/')[-1]

    main = Pipeline(args, parser, img_dir, img_path)
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

