import joblib
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from numpy.typing import NDArray
from functools import partial
from skimage.feature import multiscale_basic_features
from skimage.future import fit_segmenter, predict_segmenter
from sklearn.ensemble import RandomForestClassifier

class CustomModel():
    def __init__(self) -> None:
        self.training_image = None
        self.training_mask = None
        

    def load_training(self, train_img:'NDArray', train_mask:'NDArray'):
        self.training_image = iio.imread(train_img)
        self.training_mask = iio.imread(train_mask)

    def check_mask_labels(self) -> None:
        """
        Check whether input mask has unique values of 1, 2 and 3 only.
        """
        if not np.array_equal(np.unique(self.training_mask), [1,2,3]):
            raise ValueError('The training mask must only contain values 0, 1 and 2. Please ensure you have generated the training mask correctly (see GitHub documentation for more details).')
    
    def extract_features(self, image:'NDArray'):
        """
        Extract features from input image for Random Forest Training
        """
        features = partial(multiscale_basic_features,
                            image = image,
                            intensity=True,
                            edges=False,
                            texture=True,
                            sigma_min=1,
                            sigma_max=8,
                            channel_axis=-1)
        return features

    def train_random_forest(self, path:str):
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
        clf = fit_segmenter(self.training_mask, self.extract_features(self.training_image), clf)



    def load_model(self, path:str):

        return joblib.load(path) 

