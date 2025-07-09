import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio

from numpy.typing import NDArray
from scipy.ndimage import label
from skimage.measure import label as lb
from skimage.measure import regionprops
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import cast, Tuple
from pyroothair.root import Root

class GetParams(Root):
    def __init__(self, img_dir:str, img:str, straight_mask: 'NDArray', root_hairs: 'NDArray',
                 original_rh_mask: 'NDArray', original_bg_mask: 'NDArray') -> None:
        super().__init__(img_dir, img, straight_mask)
        self.root_hairs = root_hairs
        self.original_rh_mask = original_rh_mask
        self.original_bg_mask = original_bg_mask      
        self.horizontal_rh_list_1, self.horizontal_rh_list_2 = [], []
        self.rh_area_list_1, self.rh_area_list_2 = [], []
        self.bin_end_list_1, self.bin_end_list_2 = [], []
        self.bin_list = []
        self.avg_rhl_list, self.avg_rha_list = [], []
        self.smooth_avg_rha, self.smooth_avg_rhl = None, None
        self.smooth_1_rhl, self.smooth_2_rhl = None, None
        self.smooth_1_rha, self.smooth_2_rha = None, None
        self.min_x, self.max_x = None, None
        self.len_d, self.len_pos = None, None
        self.area_d, self.area_pos = None, None
        self.height = None
        self.pos_regions = None
        self.gradient = None
        self.rh_pixel_intensity = None
        self.bg_pixel_intensity = None

    def clean_for_regionprops_intensity(self, mask: 'NDArray', original_img: 'NDArray') -> list:
        """
        Clean original background, root and root hair masks for calculating pixel intensity.
        """
        original_mask_labelled, labs = cast(Tuple[np.ndarray, int], lb(mask, connectivity=2, return_num=True))

        props = regionprops(original_mask_labelled, intensity_image=original_img)
        max_lab = max(props, key=lambda x:x.area).label
        cleaned_mask = original_mask_labelled == max_lab
        cleaned_mask_labelled, _ = cast(Tuple[np.ndarray, int], lb(cleaned_mask, connectivity=2, return_num=True))
        cleaned_props = regionprops(cleaned_mask_labelled, intensity_image=original_img)

        return cleaned_props


    def calculate_pixel_intensity(self, length_cutoff: int, conv:int) -> None:
        """
        Calculate pixel intensity of the root hair mask and background.
        """
        self.image = self.image[:,:,0] # convert input img to grayscale

        print('...Calculating pixel intensities...')

        if length_cutoff: # slice original input image and masks for px intensity if --length-cutoff
            _, measured = self.clean_root_chunk(self.original_rh_mask) # measure original rh mask
            _, _, max_r, _ = [i.bbox for i in measured][0] # get lowest point of original rh mask
            max_height = round(length_cutoff * conv) # convert cutoff to pixels
            rh_props = self.clean_for_regionprops_intensity(self.original_rh_mask[max_height:max_r,:], self.image[max_height:max_r,:])
            bg_props = self.clean_for_regionprops_intensity(self.original_bg_mask[max_height:max_r,:], self.image[max_height:max_r,:])
        else:
            rh_props = self.clean_for_regionprops_intensity(self.original_rh_mask, self.image)
            bg_props = self.clean_for_regionprops_intensity(self.original_bg_mask, self.image)

        self.rh_pixel_intensity = float([i.intensity_mean for i in rh_props][0])
        self.bg_pixel_intensity = float([i.intensity_mean for i in bg_props][0])

    def sliding_window(self, height_bin_size: int, length_cutoff:float, conv:int) -> None:
        """
        Sliding window down root hair sections to compute data
        """
        print('...Calculating root hair parameters...')
        
        root_hair_segments = regionprops(self.root_hairs)
        root_hair_coords = [i.coords for i in root_hair_segments] # all coordinates of root hair segments
        max_root_coords =  max(np.max(root_hair_coords[0][:,0]), np.max(root_hair_coords[1][:,0]))

        if length_cutoff: # sliding window for length cutoff
            print(f'...Applying length cutoff of {length_cutoff} mm...')
            max_height = max_root_coords - round(length_cutoff * conv) # set endpoint of sliding window to length_cutoff if provided 
        
            for index, segment in enumerate(root_hair_segments): # loop over each root hair section (left and right side)
                min_row, min_col, max_row, max_col = segment.bbox # calculate binding box coords of each segment
                segment_mask = self.root_hairs[min_row:max_row, min_col:max_col] # mask each root hair segment
                # segment_mask = remove_small_objects(segment_mask, connectivity=2, min_size=200)
                
                for bin_start in range(max_height, max_root_coords, height_bin_size): # sliding window down each section
                    bin_end = bin_start + height_bin_size # calculate bin end
                    # print(bin_start, bin_end)
                    rh_segment = segment_mask[bin_start:bin_end, :] # define mask for sliding window for root hairs
                    _, rh_segment_measured = self.clean_root_chunk(rh_segment) 
                    rh_segment_area = [segment['area'] for segment in rh_segment_measured] # area of each segment
                    
                    for region in rh_segment_measured: # for each root hair section on either side of the root
                        _, min_segment_col, _, max_segment_col = region.bbox 
                        horizontal_rh_length = max_segment_col - min_segment_col 

                        if index == 0:
                            self.horizontal_rh_list_1.append(horizontal_rh_length)
                            self.rh_area_list_1.append(rh_segment_area)
                            self.bin_end_list_1.append(bin_end - max_height)
                                
                        elif index == 1:
                            self.horizontal_rh_list_2.append(horizontal_rh_length)
                            self.rh_area_list_2.append(rh_segment_area)
                            self.bin_end_list_2.append(bin_end - max_height) 
            

        else: # sliding window no length cutoff
            for index, segment in enumerate(root_hair_segments): # loop over each root hair section (left and right side)
                min_row, min_col, max_row, max_col = segment.bbox # calculate binding box coords of each segment
                segment_mask = self.root_hairs[min_row:max_row, min_col:max_col] # mask each root hair segment
                # segment_mask = remove_small_objects(segment_mask, connectivity=2, min_size=200)
                
                for bin_start in range(0, max_root_coords, height_bin_size): # sliding window down each section

                    bin_end = bin_start + height_bin_size # calculate bin end
                    rh_segment = segment_mask[bin_start:bin_end, :] # define mask for sliding window for root hairs
                    _, rh_segment_measured = self.clean_root_chunk(rh_segment) 
                    rh_segment_area = [segment['area'] for segment in rh_segment_measured] # area of each segment
                    
                    for region in rh_segment_measured: # for each root hair section on either side of the root
                        _, min_segment_col, _, max_segment_col = region.bbox 
                        horizontal_rh_length = max_segment_col - min_segment_col 

                        if index == 0:
                            self.horizontal_rh_list_1.append(horizontal_rh_length)
                            self.rh_area_list_1.append(rh_segment_area)
                            self.bin_end_list_1.append(bin_end)
                                
                        elif index == 1:
                            self.horizontal_rh_list_2.append(horizontal_rh_length)
                            self.rh_area_list_2.append(rh_segment_area)
                            self.bin_end_list_2.append(bin_end) 


    def clean_data(self) -> None:
        """
        Filter raw data by removing bottom 10% of RHL and rha for each side
        """
        min1_rhl = np.percentile(self.horizontal_rh_list_1, 10)
        min2_rhl = np.percentile(self.horizontal_rh_list_2, 10)

        min1_rha = np.percentile(self.rh_area_list_1, 10)
        min2_rha = np.percentile(self.rh_area_list_2, 10)

        self.horizontal_rh_list_1 = [0 if i <= min1_rhl else i for i in self.horizontal_rh_list_1]
        self.horizontal_rh_list_2 = [0 if i <= min2_rhl else i for i in self.horizontal_rh_list_2]

        self.rh_area_list_1 = [0 if float(i[0]) <= min1_rha else float(i[0]) for i in self.rh_area_list_1]           
        self.rh_area_list_2 = [0 if float(i[0]) <= min2_rha else float(i[0]) for i in self.rh_area_list_2]   
        

        # see if bin lists are different in length 
        if len(self.bin_end_list_1) != len(self.bin_end_list_2):
            raise ValueError(f'Bin positions differ in length for each segment, length {len(self.bin_end_list_1)} for list 1, and length {len(self.bin_end_list_2)} for list 2.')
        else:
            self.bin_list = self.bin_end_list_1

    def calibrate_data(self, conv: int) -> None:
        """
        Convert pixel data into mm via a conversion factor.
        """
        self.horizontal_rh_list_1 = [i/conv for i in self.horizontal_rh_list_1]
        self.horizontal_rh_list_2 = [i/conv for i in self.horizontal_rh_list_2]
        
        self.rh_area_list_1 = [i/(conv * conv)  for i in self.rh_area_list_1]
        self.rh_area_list_2 = [i/(conv * conv)  for i in self.rh_area_list_2]
        
        # reverse the order of bin_list to reflect distance from root tip/base of the root
        self.bin_list = [i/conv for i in self.bin_list]
        self.bin_list.reverse()
        
   
    
    # def get_metadata(self, metadata) -> str:
    #     """ 
    #     Get date and time from image metadata if available
    #     """
    #     try:
    #         exif = metadata['exif'] # get exif field from metadata
    #         decoded = exif.decode(errors='ignore') # decode from bytes to string
            
    #         date_pattern = r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}' # regex search pattern for date and time
            
    #         match = re.search(date_pattern, decoded)
        
    #         return match.group()
            
    #     except:
    #         return None
        
    def calculate_uniformity(self) -> None:
        """
        Calculate position along root with largest difference in RHL/rha between left and right sides of root hair sections
        """
        delta_length = [abs(x - y) for x, y in zip(self.horizontal_rh_list_1, self.horizontal_rh_list_2)]
        delta_area = [abs(x - y) for x, y in zip(self.rh_area_list_1, self.rh_area_list_2)]
        # get max difference for length, area, and the corresponding root position
        self.len_d, self.len_pos = max(list(zip(delta_length, self.bin_list)))
        self.area_d, self.area_pos = max(list(zip(delta_area, self.bin_list)))
        
    
    def calculate_growth(self, frac:float) -> None:
        """
        Apply lowess regreession to RHL and rha to estimate the root hair elongation zone
        """
        self.avg_rhl_list = [(x + y) / 2 for x, y in zip(self.horizontal_rh_list_1, self.horizontal_rh_list_2)]
        self.avg_rha_list = [(x + y) / 2 for x, y in zip(self.rh_area_list_1, self.rh_area_list_2)]

        # lowess regression to average list
        self.smooth_avg_rhl = lowess(self.avg_rhl_list, self.bin_list, frac=frac) # avg rhl
        self.smooth_avg_rha = lowess(self.avg_rha_list, self.bin_list, frac=frac) # avg rhl
        self.smooth_1_rhl = lowess(self.horizontal_rh_list_1, self.bin_list, frac=frac)
        self.smooth_2_rhl = lowess(self.horizontal_rh_list_2, self.bin_list, frac=frac)
        self.smooth_1_rha = lowess(self.rh_area_list_1, self.bin_list, frac=frac)
        self.smooth_2_rha = lowess(self.rh_area_list_2, self.bin_list, frac=frac)
        self.gradient = np.gradient(self.smooth_avg_rhl[:, 1], self.smooth_avg_rha[:, 0])
        self.pos_regions = self.gradient > 0 # retain regions of positive gradient (increasing RHL)

        labels, n_features = cast(Tuple[np.ndarray, int], label(self.pos_regions)) # label regions of bool array
        regions = [self.smooth_avg_rhl[labels == i] for i in range(1, n_features + 1)]
        longest_region = max(regions, key=len) # keep the longest growth region
        
        max_y_idx = np.argmax(longest_region[:, 1]) # get index max rhl
        max_y = longest_region[max_y_idx, 1] # get max rhl value
        self.max_x = longest_region[max_y_idx, 0] # get position along root corresponding to max rhl value

        min_y_idx =  np.argmin(np.abs(longest_region[:,1])) # get index of smalles positive x value
        min_y = longest_region[min_y_idx, 1]
        self.min_x = longest_region[min_y_idx, 0]

        self.growth_gradient = (max_y - min_y) / (self.max_x - self.min_x) # gradient of the region


    def generate_table(self, img_name: str, run_id:str, root_thickness: float, conv: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate table of summary parameters, and raw RHL/rha measurements for each image
        """
        assert self.max_x is not None
        assert self.rh_pixel_intensity is not None
        assert self.bg_pixel_intensity is not None
        # if datetime is None:
        #     datetime = 'NA'
        print('...Generating tables...')
        summary_df = pd.DataFrame({'Name': [img_name],
                                   'Batch_ID': [run_id],
                                   'Mean RHL (mm)': [np.mean(self.avg_rhl_list)],
                                   'Max RHL (mm)': [np.max(self.avg_rhl_list)],
                                   'Min RHL (mm)': [np.min(self.avg_rhl_list)],
                                   'Total RHA (mm2)': [sum(self.rh_area_list_1) + sum(self.rh_area_list_2)],
                                   'Max RHL Delta (mm)': [self.len_d],
                                   'Max RHL Delta Pos (mm)': [self.len_pos],
                                   'Max RHA Delta (mm2)': [self.area_d],
                                   'Max RHA Delta Pos (mm)': [self.area_pos],
                                   'Elongation Zone Distance (mm)': [self.max_x - self.min_x],
                                   'Elongation Zone Start (mm)': [self.min_x],
                                   'Elongation Zone Stop (mm)': [self.max_x],
                                   'Elongation Zone Gradient': [self.growth_gradient],
                                   'Root Thickness (mm)': [root_thickness / conv],
                                   'Root Length (mm)': [np.max(self.bin_list)],
                                   'RH Pixel Intensity Mean': self.rh_pixel_intensity,
                                   'Background Pixel Intensity Mean': self.bg_pixel_intensity,
                                   'RH:Background Pixel Ratio': self.rh_pixel_intensity / self.bg_pixel_intensity})

        raw_df = pd.DataFrame({'Name': [img_name] * len(self.bin_list),
                               'Distance From Root Tip (mm)': self.bin_list,
                               'RHL 1': self.horizontal_rh_list_1,
                               'RHL 2': self.horizontal_rh_list_2,
                               'RHA 1': self.rh_area_list_1,
                               'RHA 2': self.rh_area_list_2})   
        
        return summary_df, raw_df
    
    def plot_rhl(self, ax) -> None:
        """
        Plot root hair length relative to distance from root tip
        """
        assert self.smooth_1_rhl is not None
        assert self.smooth_2_rhl is not None

        ax.scatter(x=self.bin_list, y=self.horizontal_rh_list_1, color='darkmagenta', marker='*', alpha=0.3)
        ax.scatter(x=self.bin_list, y=self.horizontal_rh_list_2, color='lightseagreen', marker='X', alpha=0.3)
        ax.plot(self.smooth_1_rhl[:, 0], self.smooth_1_rhl[:, 1], color='darkmagenta', linewidth=4, linestyle='dashed', label='RHL 1')
        ax.plot(self.smooth_2_rhl[:, 0], self.smooth_2_rhl[:, 1], color='lightseagreen', linewidth=4, linestyle='dashdot', label='RHL 2')
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(self.horizontal_rh_list_2) * 2)
        ax.set_xlabel('Distance From Root Tip (mm)')
        ax.set_ylabel('Root Hair Length (mm)')
    
    def plot_avg_rhl(self, ax) -> None:
        """
        Plot average root hair length relative to distance from root tip
        Annotate regions of positive root hair growth and estimate elongation zone
        """
        assert self.smooth_avg_rhl is not None
        assert self.gradient is not None

        ax.fill_between(self.smooth_avg_rhl[:, 0], min(self.gradient) * 1.1, max(self.avg_rhl_list) * 2, where=self.pos_regions, color='cyan', alpha=0.15, label='RH Growth Regions')        
        ax.scatter(x=self.bin_list, y=self.avg_rhl_list, color='orangered')
        ax.plot(self.smooth_avg_rhl[:, 0], self.smooth_avg_rhl[:, 1], color='darkviolet', linewidth=3, label='Avg RHL')
        ax.plot((self.min_x, self.min_x), (-1, 10), color='royalblue', linewidth=2, linestyle='dashed', label='Primary Elongation Zone')
        ax.plot((self.max_x, self.max_x),(-1, 10), color='royalblue', linewidth=2, linestyle='dashed')
        ax.plot(self.smooth_avg_rhl[:, 0], self.gradient, color='green', alpha=0.7, linestyle='dashdot', label='Avg RH Gradient')
        
        ax.set_ylim(min(self.gradient) * 1.1, max(self.avg_rhl_list) * 2)
        ax.set_xlabel('Distance From Root Tip (mm)')
        ax.set_ylabel('Mean Root Hair Length (mm)')
        ax.legend(loc='upper right')
    
    def plot_rha(self, ax) -> None:
        """
        Plot root hair area relative to distance from root tip
        """
        assert self.smooth_1_rha is not None
        assert self.smooth_2_rha is not None

        ax.scatter(x=self.bin_list, y=self.rh_area_list_1, color='darkmagenta', marker='*', alpha=0.3)
        ax.scatter(x=self.bin_list, y=self.rh_area_list_2, color='lightseagreen', marker='X', alpha=0.3)
        ax.plot(self.smooth_1_rha[:, 0], self.smooth_1_rha[:, 1], color='darkmagenta', linewidth=4, linestyle='dashed', label='RHA 1')
        ax.plot(self.smooth_2_rha[:, 0], self.smooth_2_rha[:, 1], color='lightseagreen', linewidth=4, linestyle='dashdot', label='RHA 2')
        ax.set_ylim(0, max(self.rh_area_list_2) * 2)
        ax.set_xlabel('Distance From Root Tip (mm)')
        ax.set_ylabel(r'Root Hair Area (mm$^{2}$)')
        ax.legend(loc='upper right')
    
    def plot_avg_rha(self, ax) -> None:
        """
        Plot average root hair density relative to distance from root tip
        """
        assert self.smooth_avg_rha is not None

        ax.scatter(x=self.bin_list, y=self.avg_rha_list, color='orangered')
        ax.plot(self.smooth_avg_rha[:, 0], self.smooth_avg_rha[:, 1], color='darkviolet', linewidth=3, label='Avg rha')

        ax.set_ylim(0, max(self.avg_rha_list) * 2)
        ax.set_xlabel('Distance From Root Tip (mm)')
        ax.set_ylabel(r'Mean Root Hair Area (mm$^{2}$)')
        ax.legend(loc='upper right')
    
    def plot_summary(self, path:str, image_name: str) -> None:
        """
        Panel all summary plots together
        """
        labels = ['a', 'b', 'c', 'd']
        positions = [(0,0), (0,1), (1,0), (1,1)]

        fig, ax = plt.subplots(2,2, figsize=(12, 10))
        self.plot_rhl(ax[0,0])
        self.plot_rha(ax[0,1])
        self.plot_avg_rhl(ax[1,0])
        self.plot_avg_rha(ax[1,1])
        
        for label, pos in zip(labels, positions):
            ax[pos].annotate(label, xy=(0.05, 0.92), xycoords='axes fraction', fontweight='bold', fontsize=18)
            
        fig.suptitle(f'{image_name} Summary')
        plt.savefig(os.path.join(path, f'{image_name}_summary.png'))
