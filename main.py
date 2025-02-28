import argparse
import time
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from preprocess import Preprocess
from skeleton import Skeleton
from root import Root
from params import GetParams
from model import nnUNet
from images import ImageLoader

def parse_args():
    parser = argparse.ArgumentParser(prog='iRootHair')
    parser.add_argument('--input', help='Path to directory containing input image(s)', nargs='?', dest='img_dir', required=True)
    parser.add_argument('--adjusted_input', help='Path to an empty directory to store adjusted input image(s) for prediction.', nargs='?', dest='adjusted_img_dir')
    parser.add_argument('--masks', help='Path to directory to store model predicted mask(s)', nargs='?', dest='mask_dir', required=True)
    parser.add_argument('--model_path', help='Filepath to nnU-Net segmentation model', type=str, dest='model_path', required=True)
    parser.add_argument('--resolution', help='Bin size (pixels) for measurements along each root hair segment. Smaller bin sizes yield more data points per root (default = 20 px)', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--rhd_filt', help='Area threshold to remove small areas from area list; sets area for a particular bin to 0 when below the value (default = 180 px^2)', type=int, nargs='?', dest='area_filt', default=180)
    parser.add_argument('--rhl_filt', help='Length threshold to remove small lengths from length list; sets length for a particular bin to 0 when below the value (default = 14px)', type=int, nargs='?', dest='length_filt', default=14)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images (default = 127.5 px)', type=int, nargs='?', dest='conv', default=127.5)
    parser.add_argument('--output', help='Filepath to save data', type=str, dest='save_path', required=True)
    parser.add_argument('--plot_segmentation', help='Save model segmentation results in --output directory.', dest='show_segmentation', action='store_true')
    parser.add_argument('--plot_transformation', help='Save diagnostic plot showing root straightening in --output directory', dest='show_transformation', action='store_true')
    parser.add_argument('--plot_summary', help='Save summary plots for each input image in --output directory.', dest='show_summary', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    model = nnUNet()
    model.check_gpu() # determine which model to load depending on GPU availability
    
    start = time.perf_counter()

    for img in os.listdir(args.img_dir): # loop through all input images, modify and save with nnUNet prefix
        im_loader = ImageLoader()
        im_loader.read_images(args.img_dir, img)
        im_loader.resize_height()
        im_loader.resize_width()
        im_loader.resize_channel()
        im_loader.save_resized_image(args.adjusted_img_dir)

    summary = pd.DataFrame()
    raw = pd.DataFrame()

    if model.gpu_exists: # check if GPU exists
        model.setup_nnunet_paths() # set up nnUNet results path
        model.load_model(args.model_path) 
        model.run_inference(args.adjusted_img_dir, args.mask_dir) # generate predicted masks

        for mask_file in tqdm(os.listdir(args.mask_dir)): # loop through each predicted mask
            if mask_file.endswith('.png'):
                print(f'\n...Processing {mask_file}...')
                init_mask = iio.imread(os.path.join(args.mask_dir, mask_file))
                root_mask = (init_mask == 2)

                skeleton = Skeleton() 
                clean_root = skeleton.extract_root(root_mask)
                sk_y, sk_x = skeleton.skeletonize(clean_root)
                sk_spline, sk_height = skeleton.skeleton_params(sk_x, sk_y)
                med_x, med_y = skeleton.calc_skeleton_midline(sk_spline, sk_height)
                rotated_mask = skeleton.calc_rotation(med_x, med_y, init_mask) 

                rotated_root_mask = (rotated_mask == 2) 
                # plt.imsave(f'{mask_file.split('.')[0]}_clean_root.png',rotated_root_mask)
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


if __name__ == '__main__':
    main()
