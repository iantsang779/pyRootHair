import numpy as np
import matplotlib.pyplot as plt

from params import GetParams

class Plotter(GetParams):
    def __init__(self, image) -> None:
        super().__init__(image)

    def check_processing(self, mask, skeleton, ) -> None:
        _, ax = plt.subplots(nrows=2, ncols=2)
        ax[0,0].imshow(mask)
        ax[0,0].set_title('Mask')
        ax[0,1].imshow(skeleton)
        ax[0,1].set_title('Skeleton')
        ax[1,0].imshow(self.image)
        ax[1,1].imshow()

    
