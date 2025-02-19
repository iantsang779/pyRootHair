import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from preprocess import Preprocess
from skeleton import Skeleton
from root import Root
from params import GetParams
# from plots import Plotter

def parse_args():
    parser = argparse.ArgumentParser(prog='iRootHair')
    parser.add_argument('--input', help='Path to input image(s)', nargs='+', dest='img', required=True)
    # parser.add_argument('--sigma', help='Degree of gaussian blur (defualt = 1)', type=int,  nargs='?', dest='sigma', default=1)
    # parser.add_argument('--scale_factor', help='Scale factor to downsample images by (default = 4).', type=int, nargs='?', dest='factor', default=4)
    # parser.add_argument('--gain', help='Constant multiplier in sigmoid function for sigmoid exposure correction (default = 10).', type=int, nargs='?', dest='gain', default=10)
    parser.add_argument('--root_filt', help='Area threshold to remove non-main roots in the image (default = 50000 px^2).', type=int, nargs='?', dest='root_crit', default=50000)
    # parser.add_argument('--conn', help='Number of orthogonal hops to consider a pixel as a neighbor (default = 2)', type=int, nargs='?', dest='conn', default=2)
    parser.add_argument('--skeleton_bin_height', help='Bin size for sliding window down skeletoniozed root to calculate root midline (default = 100 px)', type=int, nargs='?', dest='skeleton_bin_height', default=100)
    parser.add_argument('--tip_padding', help='Number of pixels to pad around the root tip to split root hair segments (default = 50x)', type=int, nargs='?', dest='tip_padding', default=25)
    parser.add_argument('--remove_root_fragments', help='Size threshold to remove small fragments from pixel thresholding (default = 5000 px^2)', type=int, nargs='?', dest='crit', default=5000)
    parser.add_argument('--resolution', help='Bin size (pixels) for measurements along each root hair segment. Smaller bin sizes yield more data points per root (default = 100 px)', type=int, nargs='?', dest='height_bin_size', default=20)
    parser.add_argument('--remove_root_hair_fragments', help='Size threshold to remove small root hair fragments in each bin (default = 2000 px^2)', type=int, nargs='?', dest='min_size', default=2000)
    parser.add_argument('--mask_filt', help='Area threshold to remove non-primary root hair sections when masking each side (default = 1000 px^2).', type=int, nargs='?', dest='mask_filt', default=1000)
    parser.add_argument('--remove_small_areas', help='Area threshold to remove small areas from area list; sets area for a particular bin to 0 when below the value (default = 12.5 px^2)', type=int, nargs='?', dest='area_filt', default=12.5)
    parser.add_argument('--remove_small_lengths', help='Length threshold to remove small lengths from length list; sets length for a particular bin to 0 when below the value (default = 5 px)', type=int, nargs='?', dest='length_filt', default=14)
    parser.add_argument('--conv', help='The number of pixels corresponding to 1mm in the original input images (default = 255 px)', type=int, nargs='?', dest='conv', default=127.5)
    parser.add_argument('--gradient_bin', help='Bin size (cm) for sliding window over root to calculate root gradient (default = 1).', type=int, dest='grad_bin', default=1)
    parser.add_argument('--output', help='Filepath to save data', type=str, dest='output_path')
    parser.add_argument('--plots', help='Filepath to save plot(s).', type=str, dest='save_path')

    return parser.parse_args()



def main():
    start = time.perf_counter()
    args = parse_args()

    summary = pd.DataFrame()
    raw = pd.DataFrame()

    for img_path in tqdm(args.img):
        image = Preprocess(img_path)
        # processed_image = image.adjust_image(args.factor, args.gain, args.sigma)
        # root_mask, _ =  image.motsu_threshold(processed_image)
        root_mask, _, init_mask = image.load_mask('/home/iantsang/Images/Wheat/Raw/pic26_shrunk_Simple Segmentation.png')

        skeleton = Skeleton()
        clean_root = skeleton.extract_root(root_mask, args.root_crit)
        sk_y, sk_x = skeleton.skeletonize(clean_root)
        sk_spline, sk_height = skeleton.skeleton_params(sk_x, sk_y)
        med_x, med_y = skeleton.calc_skeleton_midline(sk_spline, sk_height, args.skeleton_bin_height)
        rotated_root_clean = skeleton.calc_rotation(med_x, med_y, clean_root)

        # rotated_root_mask, _ = image.motsu_threshold(rotated_img)
        # rotated_skeleton = Skeleton()
        sk_r_y, sk_r_x = skeleton.skeletonize(rotated_root_clean)
        sk_r_spline, sk_r_height = skeleton.skeleton_params(sk_r_x, sk_r_y)
        med_r_x, med_r_y = skeleton.calc_skeleton_midline(sk_r_spline, sk_r_height, args.skeleton_bin_height)
        skeleton.add_endpoints(med_r_x, med_r_y)
        skeleton.calc_skel_euclidean()
        skeleton.generate_buffer_coords(rotated_root_clean)
        straight_mask = skeleton.straighten_image(rotated_root_clean)
        
        rt = Root(straight_mask)

        # final_root_mask, final_rh_mask = rt.motsu_threshold(straightened_image)

        final_root = rt.check_root_tip()
        rt.find_root_tip()
        root_hairs = rt.trim_rh_mask(args.crit)
        rt.split_root_coords(root_hairs, args.tip_padding)

        data = GetParams(root_hairs)
        data.sliding_window(args.height_bin_size, args.mask_filt, args.factor, args.conn)
        data.clean_data(args.area_filt, args.length_filt)
        data.calibrate_data(args.conv, args.factor)
        data.calculate_avg_root_thickness(final_root, args.conv, args.factor, args.conn)
        data.calculate_uniformity()
        data.calculate_growth()
        datetime = data.get_metadata(image.image_metadata)
    
        summary_df, raw_df = data.generate_table(image.image_name, datetime)

        summary = pd.concat([summary_df,summary])
        raw = pd.concat([raw_df, raw])

        print(summary)
        print(raw)

    if args.output_path:
        summary.to_csv(f'{args.output_path}_summary.csv')
        raw.to_csv(f'{args.output_path}_raw.csv')
        
    print(f'Runtime for current batch of images: {time.perf_counter()-start:.2f} seconds.')


if __name__ == '__main__':
    main()
