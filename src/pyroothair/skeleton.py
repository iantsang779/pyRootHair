import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.typing import NDArray
from skimage.transform import rotate, PiecewiseAffineTransform, warp
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean
from scipy.interpolate import CubicSpline
from pyroothair.images import ImageLoader
from typing import cast, Tuple

class Skeleton(ImageLoader):
   
    def __init__(self, img_dir, img) -> None:
        super().__init__(img_dir, img)
        self.points = None
        self.new_points = None
        self.old_buffer_coords = None
        self.new_buffer_coords = None

    def clean_root_chunk(self, mask: 'NDArray') -> tuple[NDArray, list]:
        """
        Clean up each small section of the root mask by removing all but the largest area present
        """
        root_section_labeled, num_labels = cast(Tuple[np.ndarray, int], label(mask, connectivity=2, return_num=True)) # label the root mask
        
        # if num_labels > 0:
        root_section_measured = regionprops(root_section_labeled) # measure the root section 
        max_label = max(root_section_measured, key=lambda x: x.area).label # get the label associated with the largest area in the measured section
        
        # mask out the smaller sections, retaining only the largest section
        clean_root_mask = root_section_labeled == max_label 
        root_section_labeled, _ = cast(Tuple[np.ndarray, int], label(clean_root_mask, connectivity=2, return_num=True)) # re label root 
        root_section_measured = regionprops(root_section_labeled) # re measure root section

        return root_section_labeled, root_section_measured
      
        
    def extract_root(self, root_mask: 'NDArray') -> 'NDArray':
        """
        Filter out non-primary root sections from root mask
        """
        root_labeled_cleaned, root_count_cleaned = cast(Tuple[np.ndarray, int], label(root_mask, connectivity=2, return_num=True)) # re check num objects

        if root_count_cleaned > 1: # if more than 1 root is present    
            root_labeled_cleaned, count = self.clean_root_chunk(root_mask)

        return root_labeled_cleaned

    def skeletonize(self, root_clean: 'NDArray') -> tuple['NDArray', 'NDArray']:
        """
        Skeletonize root and get skeleton co-ordinates
        """

        skeleton = skeletonize(root_clean)
        skeleton_y, skeleton_x = np.nonzero(skeleton)

        return skeleton_y, skeleton_x

    def skeleton_params(self, skel_x: 'NDArray', skel_y: 'NDArray') -> tuple['NDArray', int, int]:
        """
        Fit cubic spline to root skeleton
        """

        t_range  = np.arange(len(skel_x))
        x_spline = CubicSpline(t_range, skel_x)(t_range)
        y_spline = CubicSpline(t_range, skel_y)(t_range)

        merged_spline = np.array(list(zip(x_spline, y_spline)))
        skeleton_start = int(min(y_spline))
        skeleton_end = int(max(y_spline))

        return merged_spline, skeleton_start, skeleton_end

    def calc_skeleton_midline(self, start:int, end: int, merged_spline: 'NDArray') -> tuple[list, list]:
        """
        Calculate midline of original root skeleton using a sliding window
        """
        med_x, med_y = [], []
        
        bin_size = 100
        for start in range(start, end, bin_size):
            end = start + bin_size
            bin_y_val = [x[1] for x in merged_spline if start <= x[1] <= end]
            bin_x_val = [x[0] for x in merged_spline if start <= x[1] <= end]
            
            # get x and y spline median for bin_size
            med_y.append(np.median(bin_y_val)) 
            med_x.append(np.median(bin_x_val))

        return med_x, med_y
    
    def calc_rotation(self, med_x: list, med_y: list, init_mask: 'NDArray') -> 'NDArray':
        """
        Get angle of rotation from root skeleton
        """

        dy = max(med_y) - min(med_y)
        dx = med_x[0] - med_x[-1]

        angle = np.rad2deg(np.arctan(dx/dy))
        
        rotated_mask = rotate(init_mask, angle, preserve_range=True, resize=True, mode='constant')

        # rotated_label_mask, _ = self.clean_root_chunk(rotated_mask)

        return rotated_mask
   
    def add_endpoints(self, med_x: list, med_y: list) -> None:
        """
        Add endpoints beyond image boundaries to capture ends of image during warping
        """

        self.points = np.array(list(zip(med_x, med_y)))

        first_point = self.points[0] 
        second_point = self.points[1]

        last_point = self.points[-1]
        seclast_point = self.points[-2]

        def generate_endpoints(pointA: list, pointB: list, length: int) -> tuple[float, float]:
            """
            Generate new end points
            """
            len_pApB = euclidean(pointA, pointB)
            n_x = pointA[0] + ((pointA[0]-pointB[0]) / len_pApB*length)
            n_y = pointA[1] + ((pointA[1]-pointB[1]) / len_pApB*length)

            return n_x, n_y

        first_x, first_y = generate_endpoints(first_point, second_point, 100)
        last_x, last_y = generate_endpoints(last_point, seclast_point, 100)

        # Add new end points to the beginning/end of points array
        self.points = np.vstack([[first_x, first_y], self.points])
        self.points = np.vstack([self.points,[last_x, last_y]])
    
    
    def calc_skel_euclidean(self) -> None:
        """
        Calculate co-ordinates of new points to apply transformation
        """

        dist = [euclidean(x,y) for x,y in zip(self.points, self.points[1:])] # euclidean distance between each midline point
        dist_lookup = list(zip(self.points[:-1],dist))

        old_p = self.points[0]
        new_p = old_p + (0, dist[0])
        new_points = [old_p, new_p,]

        # create new array to store co-ordinates of new points 
        for x in dist_lookup[1:]:
            last_point = new_points[-1]
            _, dist = x

            length = euclidean(old_p, last_point)
            new_y_val = last_point[1] + ((last_point[1] - old_p[1]) / length*dist)
            new_points.append([old_p[0], new_y_val]) 
        
        self.new_points = np.array(new_points)

    def generate_buffer_coords(self, rotated_mask: 'NDArray') -> None:
        """
        Generate co-ordinates to pad around the root
        Define region to be warped during transformation
        """

        padding = rotated_mask.shape[1] 
        
        self.old_buffer_coords = np.vstack([self.points+[padding,0], self.points+[-padding,0]])
        self.new_buffer_coords = np.vstack([self.new_points+[padding,0], self.new_points+[-padding,0]])
    
    def visualize_transformation(self, mask: 'NDArray', path: str, img_name: str) -> None:
        """
        Check transformation co-ordinates have been correctly mapped
        """
        _, ax = plt.subplots()
        plt.imshow(mask)
        plt.plot(self.points[:,0], self.points[:,1], 'ro', label='root midline')
        plt.plot(self.old_buffer_coords[:,0], self.old_buffer_coords[:,1], 'yx', label='original buffer coords')
        plt.plot(self.new_buffer_coords[:,0], self.new_buffer_coords[:,1], 'g+', label='new buffer coords')
        ax.legend()
        plt.savefig(os.path.join(path, img_name))


    def straighten_image(self, rotated_mask: 'NDArray') -> 'NDArray':
        """
        Use piecewise affine transformation to straighten the root
        """
        tform = PiecewiseAffineTransform()
        tform.estimate(self.new_buffer_coords,self.old_buffer_coords)

        straight_mask = warp(rotated_mask, tform, mode='symmetric')
        print('...Straightening the root...')
        return straight_mask




