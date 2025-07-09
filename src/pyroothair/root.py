import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from numpy.typing import NDArray
from typing import cast, Tuple
from pyroothair.skeleton import Skeleton

class Root(Skeleton):

    def __init__(self, img_dir:str, img:str, straight_mask: 'NDArray') -> None:
        super().__init__(img_dir, img)
        self.straight_mask = straight_mask
        self.found_tip = False
        self.root_tip_x, self.root_tip_y = None, None
        self.root_start_x, self.root_start_y = None, None
        self.final_labeled_root = None
        self.final_root_mask = (self.straight_mask > 1.7) # set final masks for root and root hair
        self.rh_mask = (self.straight_mask > 0.4) & (self.straight_mask <= 1.4) 
        self.rh_mask_labeled = None
        self.count = None
        self.root_thickness = None
        self.split_root = None
        self.error = False


    def check_root_tip(self) -> 'NDArray':
        """
        Check whether root tip is present in root mask
        
        """

        self.final_root_mask = self.extract_root(self.final_root_mask) # get rid of all but main root, should only be 1 root left
        self.final_labeled_root, _ = cast(Tuple[np.ndarray, int], label(self.final_root_mask, connectivity=2, return_num=True))

        root_measured = regionprops(self.final_labeled_root) # measure cleaned root
        coords = [i.coords for i in root_measured][0] # get all coords of masked cleaned root
        max_root_y_coord = max(coords[:,0]) # get max y-coord of cleaned root
        image_height = self.straight_mask.shape[0] # get height of the image
        
        if image_height - max_root_y_coord > 1: # if > 1 px difference between image height and max y of root
            self.found_tip = True 
        return self.final_labeled_root
    
    def calculate_avg_root_thickness(self, final_root_labeled: 'NDArray', length_cutoff: float, conv: int) -> float:
        """
        Calculate average root thickness from root mask via sliding window
        """
        width_list = []

        root_measured = regionprops(final_root_labeled)
        root_params = [i.bbox for i in root_measured]
        root_start, _, root_end, _ = root_params[0]

        if length_cutoff:
            if length_cutoff * conv > root_end:
                raise ValueError(f'Cannot specify a length cutoff ({length_cutoff}) greater than the existing root length ({root_end / conv}).')
            root_end = round(length_cutoff * conv)
            
        for start in range(root_start, root_end, 100):

            end = start + 100
            root_section = final_root_labeled[start:end, :]
            _, root_section_measured = self.clean_root_chunk(root_section) # remove any small fragments from binning
            root_binned_params =  [i.bbox for i in root_section_measured]
            _, min_col, _, max_col = root_binned_params[0] # get bounding box for min and max col of root per bin
            root_width = max_col - min_col

            width_list.append(root_width)

        self.root_thickness = float(np.mean(width_list)) # mean root thickness in px
        self.split_root = np.percentile(width_list, 20) # get 20th percentile of root thickness to use as padding

        return self.root_thickness

    def find_root_tip(self) -> None:
        """
        Find location of root tip from skeletonized root
        """

        if self.found_tip:
            # rotate root skeleton 
            final_skeleton = skeletonize(self.final_labeled_root)
            kernel = np.array([[1,1,1], 
                               [1,2,1],  # each pixel in sliding window has value of 2 (2 x 1), while neighboring pixels have a value of 1 
                               [1,1,1]]) # define kernel that slides over each pixel in the rotated root skeleton.
        
        
            neighbours = convolve(final_skeleton.astype(int), kernel, mode='constant') # apply convolution to skeleton to find out which pixels have 1 neighbour
            endpoints = np.where(neighbours == 3) # edges only have 1 neighbour, so 2 + 1 = 3
            endpoints = list(zip(endpoints[0], endpoints[1])) # store results in paired list 
            root_tip = max(endpoints, key = lambda x: x[0]) # get coords where y-coord is max (bottom of root - assuming root growing downwards)
            root_start = min(endpoints, key=lambda x: x[0]) # coords of where root starts
            self.root_tip_y, self.root_tip_x = root_tip 
            self.root_start_y, self.root_start_x = root_start
    
    def process_rh_mask(self) -> None:
        """
        Clean up root hair mask, and determine how many 'root hair chunks' are present.
        This function differentiates the main root hair segment from non-primary root hair fragments and segmentation errors.
        """
        rh_mask_labeled, rh_count = cast(Tuple[np.ndarray, int], label(self.rh_mask, connectivity=2, return_num=True))
        rh_props = regionprops(rh_mask_labeled)
        areas = [i.area for i in rh_props]
        print(f'...Found {rh_count} root hair fragments...')
    
        # here I use k-means clustering to cluster the areas list into 2 groups, root hairs, and rubbish
        # this approach is more robust than using remove_small_objects(), as the crit threshold can differ 
        # significantly for different species/user images

        # run k-means clustering if there are more than 2 rh sections
        # otherwise kmeans will ignore one of the main sections
        new_rh_mask = np.full(self.rh_mask.shape, fill_value=False) # create array of all False's of same shape as rh mask

        if rh_count > 2:
            areas_reshaped = np.reshape(areas, (-1, 1)) # reshape areas for kmeans clustering

            kmeans_areas = KMeans(n_clusters=2).fit(areas_reshaped)
            # rh_to_ignore = areas[kmeans_areas.labels_ == 0] # areas of everything else to ignore
            main_rh_area = areas_reshaped[kmeans_areas.labels_ == 1] # area of main rh segments (either connected, as 1 area, or 2 large areas)
            # print(kmeans_areas.labels_)

            for i in rh_props: # iterate over all regions
                for z in main_rh_area: # iterate over all areas in main_rh_area
                    if i.area == z:
                        new_rh_mask[rh_mask_labeled == i.label] = self.rh_mask[rh_mask_labeled == i.label] # set locations of the main root hair to True in the new mask
            print(f'...Removing secondary root hair fragments...')
            self.rh_mask = new_rh_mask
        

    def split_root_coords(self) -> 'NDArray':
        """
        Split the root hair mask around the location of root tip and root start based on mean root thickness
        """
      
        assert self.split_root is not None
      
        # self.rh_mask = self.extract_root(self.rh_mask) # keep the largest RH chunk (pre-tip splitting)

        padding = int(self.split_root // 2.65)

        if self.found_tip:
            assert self.root_start_y is not None
            assert self.root_start_x is not None
            assert self.root_tip_y is not None
            assert self.root_tip_x is not None
            root_tip_y_max, root_tip_y_min = self.root_tip_y + padding*5, self.root_tip_y - padding*2
            root_tip_x_max, root_tip_x_min = self.root_tip_x + padding, self.root_tip_x - int(padding*0.7)

            root_start_y_max, root_start_y_min = self.root_start_y + padding*2, self.root_start_y - padding*2
            root_start_x_max, root_start_x_min = self.root_start_x, self.root_start_x - int(padding//3) 
            
            if root_start_y_min <= 0:
                root_start_y_min = 0
            self.rh_mask[root_tip_y_min:root_tip_y_max, root_tip_x_min:root_tip_x_max] = False # apply coords to mask
            self.rh_mask[root_start_y_min:root_start_y_max, root_start_x_min:root_start_x_max] = False
            
        self.rh_mask_labeled, self.count = cast(Tuple[np.ndarray, int], label(self.rh_mask, connectivity=2, return_num=True))
     
        return self.rh_mask_labeled
    
    # def trim_rh_mask(self) -> 'NDArray':
    #     """
    #     Remove fragments from root hair mask, and remove any non-primary root hair masks
    #     """
    
    #     check_root_hair = regionprops(self.rh_mask_labeled) # measure area of root hair masks
        
    #     rh_regions = sorted(check_root_hair, key = lambda x: x.area, reverse=True) # sort all root hair regions in desc order by size

    #     area_1_label = rh_regions[0].label # get label of largest RH area
    #     area_2_label = rh_regions[1].label # get label of second largest RH area
        
    #     cleaned_rh = np.logical_or(self.rh_mask_labeled == area_1_label, self.rh_mask_labeled == area_2_label) # set mask to retain only primary hair sections
        
    #     self.rh_mask_labeled, _ = label(cleaned_rh, connectivity=2, return_num=True) 

    #     return self.rh_mask_labeled 

    def crop_rh_mask(self, root_hair_mask: 'NDArray') -> 'NDArray':
        """
        Crop root hair masks so that the start co-ordinate is the same for both segments.
        Important for roots that are curved at an angle, as after rotation, left and right RH segments will be non-uniform at the top.
        """
        def get_region_coords(mask) -> list:

            props = regionprops(mask)
            coords = [i.coords for i in props]
            return coords
        
        coords = get_region_coords(root_hair_mask)

        if len(coords) > 2:
            root_hair_mask = remove_small_objects(root_hair_mask, connectivity=2, min_size=500)
            coords = get_region_coords(root_hair_mask)

        print(f'...Found {len(coords)} root hair segment masks...')

        if len(coords) != 2: # if there are not 2 sections, set is_ok to False and skip to the next image.
            self.error = True
            
        if not self.error:
            crop_start = max(np.min(coords[0][:,0]), np.min(coords[1][:,0]))
            cropped_rh_mask = root_hair_mask[crop_start:,:] # crop the final root hair section to the start of the shorter root hair segment for uniform calculation
            
            coords = get_region_coords(cropped_rh_mask) # re-calculate coordinates of cropped image

            if len(coords) != 2:
                self.error = True # set error to true if root hair segments are not split properly at the top after cropping
                return np.empty([1,1])
            else:
                crop_end = min(np.max(coords[0][:,0]), np.max(coords[1][:,0])) # crop ends of root hair sections
                final_rh_mask = cropped_rh_mask[:crop_end, :]

                return final_rh_mask

        else: # if there are not 2 sections, e.g if the tip wasn't split properly

            return np.empty([1,1]) # return empty array



    
