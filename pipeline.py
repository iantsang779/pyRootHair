import matplotlib.pyplot as plt
import os
import pandas as pd

from numpy.typing import NDArray
from params import GetParams
from root import Root
from skeleton import Skeleton

class CheckArgs():
    def __init__(self, args, parser) -> None:
        self.args = args
        self.parser = parser

    def check_arguments_gpu(self) -> None:
        """
        Check necessary arguments if GPU is available
        """
        missing_args = []
            # check necessary arguments are supplied
        if self.args.img_dir is None:
            missing_args.append('--input')
        if self.args.batch_id is None:
            missing_args.append('--batch_id')
        if self.args.model_path is None:
            missing_args.append('--model_path')
        if missing_args:
            self.parser.error(f'The following arguments are required unless --no_gpu is specified: {missing_args}')

    def check_arguments_nogpu(self)-> None:
        """
        Check necessary arguments if GPU is not available
        """
        if self.args.rfc_model_path is None:
            self.parser.error('The following argument is required when specifying --no_gpu: --rfc_model_path.')
        if self.args.model_path or self.args.custom_model_path:
            self.parser.error('Invalid argument with --no_gpu! --no_gpu should only be run with --rfc_model_path and --output.')
        if self.args.img_dir is None:
            self.parser.error('Missing argument for the input image directory --input.')

    
    def check_arguments_output(self) -> None:
        """
        Check --output argument is provided
        """
        if self.args.save_path is None:
            self.parser.error('Please specify filepath to store output data with --output.')


class Pipeline(CheckArgs):
    
    def __init__(self, check_args) -> None:
        self.check_args = check_args
        self.args = check_args.args
        self.parser = check_args.parser

    def run_pipeline(self, init_mask: 'NDArray', filename:'str') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Core pipeline bringing together logic across all modules
        """
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
        rt.split_root_coords()
        root_hairs = rt.trim_rh_mask()
        
        data = GetParams(root_hairs)
        data.sliding_window(self.args.height_bin_size)
        data.clean_data(self.args.area_filt, self.args.length_filt)
        data.calibrate_data(self.args.conv)
        data.calculate_avg_root_thickness(final_root, self.args.conv)
        data.calculate_uniformity()
        data.calculate_growth()
        # datetime = data.get_metadata(image.image_metadata)
    
        summary_df, raw_df = data.generate_table(filename.split('.')[0], self.args.batch_id)


        if self.args.show_transformation:
            self.check_args.check_arguments_output()
            skeleton.visualize_transformation(init_mask, self.args.save_path, filename.split('.')[0]) 

        if self.args.show_segmentation:
            self.check_args.check_arguments_output()
            plt.imsave(os.path.join(self.args.save_path,f'{filename.split('.')[0]}_mask.png'), straight_mask)
            plt.imsave(os.path.join(self.args.save_path,f'{filename.split('.')[0]}_root_hair_mask.png'), root_hairs)
        
        if self.args.show_summary:
            self.check_args.check_arguments_output()
            data.plot_summary(self.args.save_path, filename.split('.')[0])

        return summary_df, raw_df