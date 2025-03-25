import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize
from scipy.ndimage import convolve
from numpy.typing import NDArray

from skeleton import Skeleton


class Root(Skeleton):

    def __init__(self, straight_mask: 'NDArray') -> None:
        self.straight_mask = straight_mask
        self.found_tip = False
        self.root_tip_x, self.root_tip_y = None, None
        self.final_labeled_root = None
        self.final_root_mask = (self.straight_mask > 1.5) # set final masks for root and root hair
        self.final_rh_mask = (self.straight_mask > 0.4) & (self.straight_mask <= 1.4) 
        self.final_rh_mask_labeled = None
        self.count = None

    def check_root_tip(self) -> None:
        """
        Check whether root tip is present in root mask
        
        """
        self.final_labeled_root, _ = label(self.final_root_mask, connectivity=2, return_num=True)
        root_measured = regionprops(self.final_labeled_root) # measure cleaned root
        coords = [i.coords for i in root_measured][0] # get all coords of masked cleaned root
        max_root_y_coord = max(coords[:,0]) # get max y-coord of cleaned root
        image_height = self.straight_mask.shape[0] # get height of the image
        
        if image_height - max_root_y_coord > 1: # if > 1 px difference between image height and max y of root
            self.found_tip = True 
            
        return self.final_labeled_root
    
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
            print('...Located root tip...')
            self.root_tip_y, self.root_tip_x = root_tip 
        else:
            print('...Failed to locate root tip...')        

    def split_root_coords(self) -> None:
        """
        Split the root hair mask around the location of root tip
        """
        tip_border = 20

        if self.found_tip:
            root_tip_y_max, root_tip_y_min = self.root_tip_y + tip_border, self.root_tip_y - tip_border
            root_tip_x_max, root_tip_x_min = self.root_tip_x + tip_border, self.root_tip_x - tip_border
            
            self.final_rh_mask[root_tip_y_min:root_tip_y_max, root_tip_x_min:root_tip_x_max] = False # apply coords to mask

        self.final_rh_mask_labeled, self.count = label(self.final_rh_mask, connectivity=2, return_num=True)

    def trim_rh_mask(self) -> 'NDArray':
        """
        Remove fragments from root hair mask, and remove any non-primary root hair masks
        """
    
        check_root_hair = regionprops(self.final_rh_mask_labeled) # measure area of root hair masks
        
        rh_regions = sorted(check_root_hair, key = lambda x: x.area, reverse=True) # sort all root hair regions in desc order by size

        area_1_label = rh_regions[0].label # get label of largest RH area
        area_2_label = rh_regions[1].label # get label of second largest RH area
        
        cleaned_rh = np.logical_or(self.final_rh_mask_labeled == area_1_label, self.final_rh_mask_labeled == area_2_label) # set mask to retain only primary hair sections
        
        self.final_rh_mask_labeled, _ = label(cleaned_rh, connectivity=2, return_num=True) 

        return self.final_rh_mask_labeled 

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

        crop_start = max(np.min(coords[0][:,0]), np.min(coords[1][:,0]))
        cropped_rh_mask = root_hair_mask[crop_start:,:] # crop the final root hair section to the start of the shorter root hair segment for uniform calculation
        
        coords = get_region_coords(cropped_rh_mask) # re-calculate coordinates of cropped image
        
        crop_end = min(np.max(coords[0][:,0]), np.max(coords[1][:,0])) # crop ends of root hair sections
        final_rh_mask = cropped_rh_mask[:crop_end, :]

        return final_rh_mask




        
        
   