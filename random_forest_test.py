import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import joblib

from numpy.typing import NDArray
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair Random Forest Classifier')
    parser.add_argument('--train_img', help='Path to image to train the Random Forest Classifier on.', dest='train_img_path', type=str, required=True)
    parser.add_argument('--train_mask', help='Path to binary mask corresponding to the training image.', dest='train_mask_path', type=str, required=True)
    parser.add_argument('--output', help='Save name and path for the trained Random Forest Classifier', dest='model_output_path', required=True)
    parser.add_argument('--sigma_min', help='Minimum sigma for feature extraction. Default = 1', dest='sigma_min', type=int, default=1)
    parser.add_argument('--sigma_max', help='Maximum sigma for feature extraction. Default = 3', dest='sigma_max', type=int, default=4)
    parser.add_argument('--n_estimators', help='Number of trees in the Random Forest Classifier. Default = 50', dest='n_estimators', tpye=int, default=50)
    parser.add_argument('--max_depth', help='Maximum depth of the Random Forest Classifier. Default = 10', dest='max_depth', type=int, default=1)
    parser.add_argument('--max_samples', help='Number of samples extracted from features to train each estimator. Default = 0.05.', dest='max_samples', default=0.05)
    

class ForestTrainer():
    def __init__(self, train_img_path: str, train_mask_path: str) -> None:
        self.train_img = iio.imread(train_img_path)
        self.train_mask = iio.imread(train_mask_path)
        self.func = None
        self.rfc = None

    def check_mask_class(self) -> None:
        newmask = self.train_mask.copy()

        if not np.array_equal(np.unique(self.train_mask), [1,2,3]):
            newmask[self.train_mask == 0] = 1
            newmask[self.train_mask  == 1] = 2
            newmask[self.train_mask  == 2] = 3

            self.train_mask = newmask
    
    def features_func(self, sigma_min: int, sigma_max: int):
        self.func = partial(
            feature.multiscale_basic_features,
            intensity=True,
            edges=False,
            texture=True,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=-1,
        )

    def train(self, features: 'NDArray', mask: 'NDArray', 
            n_estimators:int, max_depth:int, max_samples:int, model_path):
        """
        Train a Random Forest Classsifier on a representative example of an image.
        """
        rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, max_depth=max_depth, max_samples=max_samples)
        train_features = features[mask]
        train_labels = mask.ravel()
        rfc.fit(train_features, train_labels)

        self.rfc = rfc        
        joblib.dump(self.rfc, model_path)



