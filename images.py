import imageio.v3 as iio
import magic
import numpy as np
import os

from skimage.util import img_as_ubyte
from skimage.transform import resize

class ImageLoader():

    def __init__(self) -> None:
        self.target_h, self.target_w, self.target_c = (1520, 2028, 3)
        self.old_h, self.old_w, self.old_c = (None, None, None)
        self.resized_h, self.resized_w, self.resized_c = (None, None, None)
        self.adjust_height, self.adjust_channel = (False, False)
        self.image = None
        self.resized_img = None

    def read_images(self, img_dir:str, img:str) -> None:
        """
        Read an image in user specified input directory and check dimensions.
        Check whether image is a PNG file.
        Check if with of input image is too large relative to height.
        """
        # check each input image is a PNG 
        if magic.from_file(os.path.join(img_dir, img), mime=True) != 'image/png':
            raise TypeError(f'Incorrect file format for {img}. Input images must of PNG format.')

        self.image = iio.imread(os.path.join(img_dir, img))
        print(f'\n...Loading {img}...\n')
        self.old_h, self.old_w, self.old_c = self.image.shape

        if self.old_h != self.target_h:
            self.adjust_height = True
        if self.old_c > self.target_c:
            self.adjust_channel = True

        # if original image width <= (2028/1520) * original image height
        if not self.old_w <= (self.target_w / self.target_h) * self.old_h:
            raise ValueError(f'The width of image {img} is too wide. Please crop this image, and try again.')
        
    def resize_height(self) -> None:
        """
        Resize input image such that ouput reiszed image has height of 1520 px. Resizes width by the same scale factor
        """
        if self.adjust_height:
            resize_factor = self.old_h / self.target_h
            self.resized_img = resize(self.image, (self.old_h // resize_factor), (self.old_w // resize_factor), anti_aliasing=True)

    def resize_width(self) -> None:
        """
        Pad width of image if the resized image width < 2028
        """

        _, resized_w, _ = self.resized_img.shape

        if resized_w < self.target_w: 
            pad_amount = (self.target_w - resized_w) // 2
            self.resized_img = np.pad(self.resized_img, ((0,0), (pad_amount, pad_amount+1), (0,0)), mode='median')

    def resize_channel(self) -> None:
        """
        Remove alpha channel if present
        """
        if self.adjust_channel:
            self.resized_img = self.resized_img[:,:,3]

    def save_resized_image(self, img_dir:str) -> None:
        """
        Convert image from float64 to uint8 and save image as XXX_resized.png
        """
        self.resized_img = img_as_ubyte(self.resized_img)
        new_img_name = os.path.basename(self.image).split('.')[0]
        print(f'\n...Saving resized image: {new_img_name}, in {img_dir}...\n')

        iio.imwrite(f'os.path.join({img_dir}, {new_img_name}_resized.png)', self.resized_img)






    


