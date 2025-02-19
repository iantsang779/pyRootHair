import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from itertools import zip_longest
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from statsmodels.nonparametric.smoothers_lowess import lowess
from root import Root

class GetParams(Root):
    def __init__(self, root_hairs: 'NDArray') -> None:
        self.root_hairs = root_hairs
        self.height = None
        self.horizontal_rh_list_1, self.horizontal_rh_list_2 = [], []
        self.rh_area_list_1, self.rh_area_list_2 = [], []
        self.bin_end_list_1, self.bin_end_list_2 = [], []
        self.bin_list = []
        self.avg_rhl_list, self.avg_rhd_list = [], []
        self.smooth_avg_rhd, self.smooth_avg_rhl = None, None
        self.min_x, self.max_x = None, None
        self.growth_gradient = None
        self.root_thickness = None
        self.len_d, self.len_pos = None, None
        self.area_d, self.area_pos = None, None

    def sliding_window(self, height_bin_size: int, mask_filt: int, factor: int, conn: int) -> None:
        """
        Sliding window down root hair sections to compute data
        """
        height_bin_size = height_bin_size // factor
        mask_filt = mask_filt // factor
        
        root_hair_segments = regionprops(self.root_hairs)
        for index, segment in enumerate(root_hair_segments): # loop over each root hair section (2)

            min_row, min_col, max_row, max_col = segment.bbox # calculate binding box coords of each segment
            segment_mask = self.root_hairs[min_row:max_row, min_col:max_col] # mask each root hair segment]
            segment_mask = remove_small_objects(segment_mask, min_size=mask_filt)
            height = segment_mask.shape[0] # get height of mask
            fig, ax = plt.subplots()
            ax.imshow(segment_mask)
            for bin_start in range(0, height, height_bin_size): # sliding window down each section

                bin_end = bin_start + height_bin_size # calculate bin end
                rh_segment = segment_mask[bin_start:bin_end, :] # define mask for sliding window for root hairs
                _, rh_segment_measured = self.clean_root_chunk(rh_segment) 
                rh_segment_area = [segment['area'] for segment in rh_segment_measured] # area of each segment

                for region in rh_segment_measured: # for each root hair section on either side of the root
                    _, min_segment_col, _, max_segment_col = region.bbox 
                    horizontal_rh_length = max_segment_col - min_segment_col 

                    if index == 0:
                        # _, centroid_x1 = rh_segment_centroid 
                        self.horizontal_rh_list_1.append(horizontal_rh_length)
                        self.rh_area_list_1.append(rh_segment_area)
                        self.bin_end_list_1.append(bin_end)
                           
                    elif index == 1:
                        # _, centroid_x2 = rh_segment_centroid
                        self.horizontal_rh_list_2.append(horizontal_rh_length)
                        self.rh_area_list_2.append(rh_segment_area)
                        self.bin_end_list_2.append(bin_end) 


    def clean_data(self, area_filt: float, length_filt: float) -> None:
        """
        Filter raw data and pad lists
        """

        self.horizontal_rh_list_1 = [0 if i < length_filt else i for i in self.horizontal_rh_list_1]
        self.horizontal_rh_list_2 = [0 if i < length_filt else i for i in self.horizontal_rh_list_2]

        self.rh_area_list_1 = [0 if float(i[0]) < area_filt else float(i[0]) for i in self.rh_area_list_1]           
        self.rh_area_list_2 = [0 if float(i[0]) < area_filt else float(i[0]) for i in self.rh_area_list_2]   
        
        # see if bin lists are different in length 
        if len(self.bin_end_list_1) != len(self.bin_end_list_2):
            if len(self.bin_end_list_1) > len(self.bin_end_list_2):
                self.bin_list = self.bin_end_list_1 # set bin_list as length of longer list
            else:
                self.bin_list = self.bin_end_list_2
        else:
            self.bin_list = self.bin_end_list_1

        # pad lists together, set Nones to 0s, and unpack back into lists
        pad_lists = list(zip_longest(self.horizontal_rh_list_1, self.horizontal_rh_list_2, self.rh_area_list_1, self.rh_area_list_2, self.bin_list))
        # set Nones to 0s in pad_lists
        pad_lists = [tuple(0 if x is None else x for x in tup) for tup in pad_lists]
        # unpack tuples back into lists for each, now all of equal length 
        self.horizontal_rh_list_1, self.horizontal_rh_list_2, self.rh_area_list_1, self.rh_area_list_2, self.bin_list = map(list, zip(*pad_lists))

    def calibrate_data(self, conv: int) -> None:
        """
        Convert pixel data into mm via a conversion factor.
        """

        def _check_zeros(value, conv) -> float: # helper function to avoid dividing by zero 
            return value / conv if value != 0 else 0
        
        self.horizontal_rh_list_1 = [_check_zeros(i, conv) for i in self.horizontal_rh_list_1]
        self.horizontal_rh_list_2 = [_check_zeros(i, conv) for i in self.horizontal_rh_list_2]
        
        self.rh_area_list_1 = [_check_zeros(i, conv * conv) for i in self.rh_area_list_1]
        self.rh_area_list_2 = [_check_zeros(i, conv * conv) for i in self.rh_area_list_2]
        
        # reverse the order of bin_list to reflect distance from root tip/base of the root
        self.bin_list = [_check_zeros(i, conv) for i in self.bin_list]
        self.bin_list.reverse()
        
    def calculate_avg_root_thickness(self, final_root_labeled: 'NDArray', conv: int, factor: int, conn: int) -> None:
        """
        Calculate average root thickness from root mask via sliding window
        """
        width_list = []

        conv = conv // factor

        root_measured = regionprops(final_root_labeled)

        root_params = [i.bbox for i in root_measured]

        root_start, _, root_end, _ = root_params[0]

        for start in range(root_start, root_end, 100):

            end = start + 100
            root_section = final_root_labeled[start:end, :]
            _, root_section_measured = self.clean_root_chunk(root_section, conn) # remove any small fragments from binning
            root_binned_params =  [i.bbox for i in root_section_measured]
            _, min_col, _, max_col = root_binned_params[0] # get bounding box for min and max col of root per bin
            root_width = max_col - min_col

            width_list.append(root_width)

        self.root_thickness = np.mean(width_list) / conv

    
    def get_metadata(self, metadata) -> str:
        """ 
        Get date and time from image metadata if available
        """
        try:
            exif = metadata['exif'] # get exif field from metadata
            decoded = exif.decode(errors='ignore') # decode from bytes to string
            
            date_pattern = r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}' # regex search pattern for date and time
            
            match = re.search(date_pattern, decoded)
        
            return match.group()
            
        except:
            return None
        
    def calculate_uniformity(self) -> None:
        """
        Calculate position along root with largest difference in RHL/RHD between left and right sides of root hair sections
        """
        delta_length = [abs(x - y) for x, y in zip(self.horizontal_rh_list_1, self.horizontal_rh_list_2)]
        delta_area = [abs(x - y) for x, y in zip(self.rh_area_list_1, self.rh_area_list_2)]
        # get max difference for length, area, and the corresponding root position
        self.len_d, self.len_pos = max(list(zip(delta_length, self.bin_list)))
        self.area_d, self.area_pos = max(list(zip(delta_area, self.bin_list)))
        
    
    def calculate_growth(self) -> None:
        """
        Apply lowess regreession to RHL and RHD to estimate the root hair elongation zone
        """
        self.avg_rhl_list = [(x + y) / 2 for x, y in zip(self.horizontal_rh_list_1, self.horizontal_rh_list_2)]
        self.avg_rhd_list = [(x + y) / 2 for x, y in zip(self.rh_area_list_1, self.rh_area_list_2)]

        # lowess regression to average list
        self.smooth_avg_rhl = lowess(self.avg_rhl_list, self.bin_list, frac=0.15) # avg rhl
        self.smooth_avg_rhd = lowess(self.avg_rhd_list, self.bin_list, frac=0.15) # avg rhl

        gradient = np.gradient(self.smooth_avg_rhl[:, 1], self.smooth_avg_rhd[:, 0])
        pos_regions = gradient > 0 # retain regions of positive gradient (increasing RHL)

        labels, n_features = label(pos_regions) # label regions of bool array
        regions = [self.smooth_avg_rhl[labels == i] for i in range(1, n_features + 1)]
        longest_region = max(regions, key=len) # keep the longest growth region
        
        max_y_idx = np.argmax(longest_region[:, 1]) # get index max rhl
        max_y = longest_region[max_y_idx, 1] # get max rhl value
        self.max_x = longest_region[max_y_idx, 0] # get position along root corresponding to max rhl value

        min_y_idx = np.argmin(np.abs(longest_region[:, 1])) # same as above but get index where abs difference is closest to 0
        min_y = longest_region[min_y_idx, 1]
        self.min_x = longest_region[min_y_idx, 0]

        self.growth_gradient = (max_y - min_y) / (self.max_x - self.min_x) # gradient of the region


    def generate_table(self, img_name: str, datetime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate table of summary parameters, and raw RHL/RHD measurements for each image
        """
       
        if datetime is None:
            datetime = 'NA'

        summary_df = pd.DataFrame({'Name': [img_name],
                                #    'Mode': [self.image_metadata['mode']],
                                #    'Shape': [self.image_metadata['shape']],
                                   'DateTime': [datetime],
                                   'Avg RHL (mm)': [np.mean(self.avg_rhl_list)],
                                   'Max RHL (mm)': [np.max(self.avg_rhl_list)],
                                   'Min RHL (mm)': [np.min(self.avg_rhl_list)],
                                   r'Total RHD (mm^{2})': [sum(self.rh_area_list_1) + sum(self.rh_area_list_2)],
                                   'Max RHL Segment Delta (mm)': [self.len_d],
                                   'Max RHL Segment Pos (mm)': [self.len_pos],
                                   'Max RHD Segment Delta (mm)': [self.area_d],
                                   'Max RHD Segment Pos (mm)': [self.area_pos],
                                   'Elongation Zone Distance (mm)': [self.max_x - self.min_x],
                                   'Elongation Zone Start (mm)': [self.min_x],
                                   'Elongation Zone Stop (mm)': [self.max_x],
                                   'Elongation Zone Gradient': [self.growth_gradient],
                                   'Root Thickness (mm)': [self.root_thickness],
                                   'Root Length (mm)': [np.max(self.bin_list)]})

        raw_df = pd.DataFrame({'Name': [img_name] * len(self.bin_list),
                               'Distance From Root Tip (mm)': self.bin_list,
                               'RHL 1': self.horizontal_rh_list_1,
                               'RHL 2': self.horizontal_rh_list_2,
                               'RHD 1': self.rh_area_list_1,
                               'RHD 2': self.rh_area_list_2})   
        
        return summary_df, raw_df