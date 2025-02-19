import os
import imageio.v3 as iio

from numpy.typing import NDArray
from skimage import io, color, transform, exposure, filters


class Preprocess():
    """
    Preprocess input image.
    Grayscale, compress image size, adjust contrast and apply gaussian blur
    """
    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.image = io.imread(image_path)
        self.image_name = os.path.basename(self.image_path).split('.')[0]
        self.image_metadata = iio.immeta(uri=self.image_path)
 
    def adjust_image(self, factor:int, gain:int, sigma: int) -> 'NDArray':
        """
        Grayscale, downsize, adjust contrast and apply gaussian blur to image
        """
        self.image = color.rgb2gray(self.image)
        self.image = transform.resize(self.image, (self.image.shape[0] // factor, self.image.shape[1] // factor), anti_aliasing=True)
        self.image = exposure.adjust_sigmoid(self.image, gain=gain)
        self.image = filters.gaussian(self.image, sigma=sigma)

        return self.image

    def rotate_image(self, angle: float) -> 'NDArray':
        """
        Rotate image based on angle derived from root skeleton
        """
        self.image = transform.rotate(self.image, angle, preserve_range=True, mode='symmetric')
        
        return self.image

    def motsu_threshold(self, image) -> tuple['NDArray', 'NDArray']:
        """
        Apply multi otsu thresholding to the image
        """
        thresholds = filters.threshold_multiotsu(image)
        root_t, rh_t = 0.75*thresholds[0], thresholds[1] # adjust threshold for root
        root_mask = image < root_t
        rh_mask = (image < rh_t) & (image > root_t)
                
        return root_mask, rh_mask  
    
    def load_mask(self, mask_path: str):

        mask = iio.imread(mask_path)
        root_mask = (mask == 2)
        rh_mask = (mask == 1)

        return root_mask, rh_mask, mask
