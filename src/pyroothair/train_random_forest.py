import argparse
import numpy as np
import imageio.v3 as iio
import joblib
import os

from numpy.typing import NDArray
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from pyroothair.pipeline import CheckArgs

description = '''
Thank you for using pyRootHair!
-------------------------------

Train a RFC segmentation model.

Please read the tutorial documentation on the github repository: https://github.com/iantsang779/pyRootHair

Basic usage: pyroothair_train_random_forest --train-img /path/to/representative/training/image/example --train-mask /path/to/generated/binary/mask --model-output /path/to/output/rf_model/

Please cite the following paper when using pyRootHair: xxxxxx

Author: Ian Tsang
Contact: ian.tsang@niab.com
''' 

def parse_args():
    parser = argparse.ArgumentParser(prog='pyRootHair Random Forest Classifier',
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--train-img', help='Path to RGB image to train the Random Forest Classifier on.', dest='train_img_path', type=str, required=True)
    parser.add_argument('--train-mask', help='Path to binary mask corresponding to the training image.', dest='train_mask_path', type=str, required=True)
    parser.add_argument('--model-output', help='Save name and path for the trained Random Forest Classifier', dest='model_output_path', required=True)
    parser.add_argument('--sigma-min', help='Minimum sigma for feature extraction. Default = 1', dest='sigma_min', type=int, default=1)
    parser.add_argument('--sigma-max', help='Maximum sigma for feature extraction. Default = 4', dest='sigma_max', type=int, default=4)
    parser.add_argument('--n-estimators', help='Number of trees in the Random Forest Classifier. Default = 50', dest='n_estimators', type=int, default=50)
    parser.add_argument('--max-depth', help='Maximum depth of the Random Forest Classifier. Default = 10', dest='max_depth', type=int, default=10)
    parser.add_argument('--max-samples', help='Number of samples extracted from features to train each estimator. Default = 0.05.', dest='max_samples', default=0.05)
    
    return parser.parse_args(), parser

class ForestTrainer():

    def __init__(self) -> None:
        self.train_img = None
        self.train_mask = None
        self.func = None
        self.rfc = None

    
    def load_training(self, train_img_path:str, train_mask_path:str) -> None:
        """
        Load training image and corresponding mask
        """
        self.train_img = iio.imread(train_img_path)
        if self.train_img.shape[2] == 4:
            self.train_img = self.train_img[:,:,:3]
        self.train_mask = iio.imread(train_mask_path)

    def check_mask_class(self) -> None:
        """
        Check the unique values in the mask == 1, 2, 3, convert classes if necessary.
        1 == Background
        2 == Root Hairs
        3 == Root
        """
        assert self.train_mask is not None
        newmask = self.train_mask.copy()

        if not np.array_equal(np.unique(self.train_mask), [1,2,3]):
            # print('...Reconverting mask classes...')

            newmask[self.train_mask == 0] = 1
            newmask[self.train_mask  == 1] = 2
            newmask[self.train_mask  == 2] = 3

        self.train_mask = newmask
    
    def reconvert_mask_class(self, altered_mask:'NDArray') -> 'NDArray':
        """
        Convert class values 1,2,3 in the RFC output mask back to 0,1,2. 
        This is required for downstream pyRootHair processing
        """
        original_mask = altered_mask.copy()

        if not np.array_equal(np.unique(altered_mask), [0,1,2]):
            # print('...Reconverting mask classes...')

            original_mask[altered_mask == 1] = 0
            original_mask[altered_mask == 2] = 1
            original_mask[altered_mask == 3] = 2
            
        return original_mask

    def features_func(self, image: 'NDArray', sigma_min:int, sigma_max:int):
        """
        Extract pixel intensity, local gradients, eigenvalues of Hesssian matrix from the training image across different sigmas.

        https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/feature/_basic_features.py#L115-L198
        """
        # print('...Extracting features...')
        return multiscale_basic_features(
            image,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=-1,
        )

    def train(self, sigma_min: int, sigma_max:int, n_estimators:int, max_depth:int, max_samples:int, model_path:str) -> None:
        """
        Train a Random Forest Classifier on a representative example of an image.

        https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/future/trainable_segmentation.py#L90-L119
        """
        assert self.train_img is not None
        assert self.train_mask is not None
        
        rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, max_depth=max_depth, max_samples=max_samples)

        train_features = self.features_func(self.train_img, sigma_min, sigma_max) # extract features from image
        
        train_features_filt = train_features[self.train_mask > 0]
        train_labels = self.train_mask.ravel()

        rfc.fit(train_features_filt, train_labels)
        self.rfc = rfc        

        print('\n...Saving trained random forest model...')
        joblib.dump(self.rfc, f'{model_path}.joblib')
        print(f'\n...RFC model saved as {model_path}.joblib')

    def load_model(self, model_path:str) -> RandomForestClassifier: 
        """
        Load trained RFC segmentation model
        """
        return joblib.load(model_path)


    def predict(self, img_path:str, image:str, sigma_min:int, sigma_max:int, model) -> 'NDArray':
        """
        Predict binary mask for a given input image based on the previously trained RFC.
        
        https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/future/trainable_segmentation.py#L122-L164
        """

        img = iio.imread(os.path.join(img_path, image))
        if img.shape[2] == 4:
            img = img[:,:,:3]

        features = self.features_func(img, sigma_min, sigma_max)
        
        shape = features.shape

        if features.ndim > 2:
            features = features.reshape((-1, shape[-1]))

        predicted_labels = model.predict(features)
        output = predicted_labels.reshape(shape[:-1])
        return output
    
def main():
    args, parser = parse_args()
    check_args = CheckArgs(args, parser)

    check_args.check_arguments_train_rfc()
    rf = ForestTrainer()
    rf.load_training(args.train_img_path, args.train_mask_path)
    rf.check_mask_class()
    rf.train(args.sigma_min, args.sigma_max, args.n_estimators, args.max_depth, args.max_samples, args.model_output_path)
    

if __name__ == '__main__':
    main()


