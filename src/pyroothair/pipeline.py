import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from pathlib import Path
from numpy.typing import NDArray
from pyroothair.params import GetParams
from pyroothair.root import Root
from pyroothair.skeleton import Skeleton

class CheckArgs():
    """
    Check command line arguments for various pipeline configurations.
    """
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
            missing_args.append('-i/--input')
        if self.args.batch_id is None:
            missing_args.append('-b/--batch_id')
        if self.args.save_path is None:
            missing_args.append('-o/--output')
        if missing_args:
            self.parser.error(f'The following arguments are required when running pyRootHair using the main pipeline: {missing_args}')

    def check_arguments_rfc(self)-> None:
        """
        Check necessary arguments if GPU is not available
        """

        if self.args.rfc_model_path is None:
            self.parser.error('The following argument is required if using a Random Forest Classifier: --rfc_model_path.')
        if self.args.img_dir is None:
            self.parser.error('Missing argument for the input image directory -i/--input.')
        if self.args.input_mask:
            self.parser.error('Conflicting arguments: --input_mask is not compatible with --rfc_model_path.')
        if self.args.batch_id is None:
            self.parser.error('Missing argument for batch_id: -b/--batch_id')
        if self.args.save_path is None:
            self.parser.error('Missing argument for output: -o/--output')

    def check_arguments_output(self) -> None:
        """
        Check --output argument is provided
        """
        if self.args.save_path is None:
            self.parser.error('Please specify filepath to store output data with -o/--output.')


    def check_arguments_single_mask(self) -> None:
        """
        Check necessary arguments if processing a single binary mask
        """
        missing_args = []
        if self.args.rfc_model_path:
            self.parser.error('Conflicting arguments: --input_mask was specified, which is not compatible with --rfc_model_path.')
        if self.args.img_dir:
            self.parser.error('Conflicting arguments: -i/--input is for a directory containing a batch of images intended for batch processing. Please use --input_mask instead for a single mask.')
        if self.args.save_path is None:
            missing_args.append('-o/--output')
        if self.args.batch_id is None:
            missing_args.append('-b/--batch_id')
        if self.args.input_mask is None:
            missing_args.append('--input_mask')
        if missing_args:
            self.parser.error(f'The following arguments are required when using `-p single` for processing an individual mask: {missing_args}')
    
    def check_argument_demo(self) -> None:
        """
        Check necessary arguments for pyroothair_run_demo
        """
        missing_args = []
        
        if self.args.batch_id is None:
            missing_args.append('-b/--batch_id')
        if self.args.save_path is None:
            missing_args.append('-o/--output')
        if missing_args:
            self.parser.error(f'The following arguments are required when running pyroothair_run_demo: {missing_args}')
        

    def convert_mask(self, mask: 'NDArray') -> 'NDArray':
        """
        Convert pixel classes in an ilastik generated segmentation mask
        """
        newmask = mask.copy()

        if not np.array_equal(np.unique(mask), [0,1,2]):
            print(f'\n...Converting Mask Classes from {np.unique(mask)} to {0,1,2} ...')

            newmask[mask == 1] = 0
            newmask[mask  == 2] = 1
            newmask[mask  == 3] = 2
        
        return newmask

class Pipeline(CheckArgs):
    """
    Run core pipeline to mine traits from binary masks generated from inference
    """
    def __init__(self, check_args) -> None:
        self.check_args = check_args
        self.args = check_args.args
        self.parser = check_args.parser
        if self.args.save_path:
            if not Path(self.args.save_path).exists():
                    Path(self.args.save_path).mkdir(parents=True, exist_ok=True)
        self.plots_path = Path(self.args.save_path) / 'plots' / self.args.batch_id
        self.plots_path.mkdir(exist_ok=True, parents=True)

    def run_pipeline(self, init_mask: 'NDArray', filename:'str') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Core pipeline bringing together logic across all modules
        """
        root_mask = (init_mask == 2)

        skeleton = Skeleton() 
        clean_root = skeleton.extract_root(root_mask)
        sk_y, sk_x = skeleton.skeletonize(clean_root)
        sk_spline, sk_start, sk_end = skeleton.skeleton_params(sk_x, sk_y)
        med_x, med_y = skeleton.calc_skeleton_midline(sk_start, sk_end, sk_spline)
        rotated_root_mask = skeleton.calc_rotation(med_x, med_y, root_mask) 
        rotated_mask = skeleton.calc_rotation(med_x, med_y, init_mask)
        
        clean_root_rotated = skeleton.extract_root(rotated_root_mask)
        sk_r_y, sk_r_x = skeleton.skeletonize(clean_root_rotated)
        sk_r_spline, sk_r_start, sk_r_end = skeleton.skeleton_params(sk_r_x, sk_r_y)
        med_r_x, med_r_y = skeleton.calc_skeleton_midline(sk_r_start, sk_r_end, sk_r_spline)
        skeleton.add_endpoints(med_r_x, med_r_y)
        skeleton.calc_skel_euclidean()
        skeleton.generate_buffer_coords(rotated_mask)
        straight_mask = skeleton.straighten_image(rotated_mask)

        rt = Root(straight_mask)
        final_root = rt.check_root_tip()
        root_thickness = rt.calculate_avg_root_thickness(final_root)
        rt.find_root_tip()
        rt.process_rh_mask()
        root_hairs = rt.split_root_coords()
        root_hairs_cropped = rt.crop_rh_mask(root_hairs)

        if root_hairs_cropped.shape != (1,1):
            data = GetParams(root_hairs_cropped)
            data.sliding_window(self.args.height_bin_size)
            data.clean_data()
            data.calibrate_data(self.args.conv)
            data.calculate_uniformity()
            data.calculate_growth(self.args.frac)
    
            summary_df, raw_df = data.generate_table(filename.split('.')[0], self.args.batch_id, root_thickness, self.args.conv)

            if self.args.show_summary:
                self.check_args.check_arguments_output()
                data.plot_summary(self.plots_path, filename.split('.')[0])

            if self.args.show_transformation:
                self.check_args.check_arguments_output()
                skeleton.visualize_transformation(rotated_mask, self.plots_path, filename.split('.')[0]) 

            if self.args.show_segmentation:
                self.check_args.check_arguments_output()
                plt.imsave(os.path.join(self.plots_path,f'{filename.split('.')[0]}_mask.png'), straight_mask)
                plt.imsave(os.path.join(self.plots_path,f'{filename.split('.')[0]}_root_hair_mask_cropped.png'), root_hairs_cropped)

            return summary_df, raw_df

        else:
            if self.args.show_segmentation:
                self.check_args.check_arguments_output()
                plt.imsave(os.path.join(self.plots_path,f'{filename.split('.')[0]}_mask.png'), straight_mask)
                plt.imsave(os.path.join(self.plots_path,f'{filename.split('.')[0]}_root_hair_mask.png'), root_hairs)

            return [], []
