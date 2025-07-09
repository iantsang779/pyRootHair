import imageio.v3 as iio
# import magic
import numpy as np
import os

from skimage.util import img_as_ubyte
from skimage.transform import resize
from pathlib import Path

class ImageLoader():

    def __init__(self, img_dir:str, img:str) -> None:
        self.old_h, self.old_w, self.old_c = (None, None, None)
        self.adjust_height, self.adjust_channel = (False, False)
        self.image = iio.imread(os.path.join(img_dir, img))
        self.image_name = img
        self.sub_dir_path = None
        self.old_h, self.old_w, self.old_c = self.image.shape
        if self.old_c > 3:
            self.adjust_channel = True

    # def read_images(self) -> None:
    #     """
    #     Read an image in user specified input directory and check dimensions.
    #     Check whether image is a PNG file.
    #     Check if with of input image is too large relative to height.
    #     """
    #     # check each input image is a PNG 
    #     # if magic.from_file(os.path.join(img_dir, img), mime=True) != 'image/png':
    #     #     raise TypeError(f'Incorrect file format for {img}. Image must be a PNG!')

    #     # self.image = iio.imread(os.path.join(img_dir, img))
    #     # self.image_name = img
        

    #     # if self.old_h > 5000:
    #     #     self.adjust_height = True
        

        
    # def resize_image(self) -> None:
    #     """
    #     Resize input image if height > 5000px
    #     """
    #     assert self.old_h is not None
    #     assert self.old_w is not None

    #     if self.adjust_height:
            
    #         self.image = resize(self.image, (int(round(self.old_h / 3)), int(round(self.old_w / 3))), anti_aliasing=True)


    def resize_channel(self) -> None:
        """
        Remove alpha channel if present
        """
        assert self.image is not None

        if self.adjust_channel:
            self.image = self.image[:,:,:3]

    def setup_dir(self, out_dir: str, run_id:str) -> None:
        """ 
        Setup folders such that renamed images are in .../output/adjusted_images/batch_id
        """

        # input_path = Path(img_dir) # path of the input image directory
        # parent_dir = input_path.parent # get parent of the image directory

        output_path = Path(out_dir)
        adjusted_dir = output_path / 'adjusted_images'
        adjusted_dir.mkdir(parents=True, exist_ok=True) # make dir to store adjusted images if it doesn't exist
        
        sub_dir = adjusted_dir / run_id
        sub_dir.mkdir(parents=True, exist_ok=True) # make sub dir within adjusted_images with the user specified run_id
        self.sub_dir_path = Path(sub_dir)

    def save_resized_image(self) -> None:
        """
        Store adjusted images in a new directory
        Convert image from float64 to uint8 and save image as XXX_resized.png
        Save images with _0000.png suffix for nnUNet 
        """
        assert self.image_name is not None
        assert self.image is not None
        assert self.sub_dir_path is not None

        x = self.image_name.split('.')

        img_name = f'{'_'.join(x[:-1])}.{x[-1]}' if len(x) > 2 else x[0] # handle if user has '.' separating words in image name

        if self.adjust_height or self.adjust_channel:
            
            self.image = img_as_ubyte(self.image)

        if not img_name.endswith('_0000.png'):
            new_img_name = os.path.join(self.sub_dir_path, f'{img_name}_0000.png')

            if not Path(new_img_name).exists(): # check if renamed image exists to avoid re-saving
                iio.imwrite(new_img_name, self.image)
                print(f'\n...Renaming image {img_name} to {img_name}_0000.png in {self.sub_dir_path} for inference...\n')

        else: # if images already have _0000 suffix, save them in 
            iio.imwrite(os.path.join(self.sub_dir_path, img_name), self.image)
            print(f'\n...Saving a copy of {img_name} in {self.sub_dir_path} for inference...\n')

