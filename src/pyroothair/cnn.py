import torch
import os
import requests

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class nnUNetv2():

    def __init__(self, in_dir:str, run_id:str):
        self.in_dir = in_dir # user input directory containing raw images
        self.run_id = run_id
        self.predictor=None
        self.model_path = os.path.join(Path(__file__).parent, 'model')

        print('#########################################')
        print('     Thank you for using pyRootHair!     ')
        print('#########################################\n')  

    def check_gpu(self) -> None:
        """
        Check whether GPU is available on local machine/cluster
        """
        
        if torch.cuda.is_available():
            self.gpu_exists = True
            print(f'\n...GPU Detected! Using {torch.cuda.get_device_name(0)}...\n')
        else:
            print(f'\n...No GPU Detected...\n')

    def download_model(self) -> None:
        
        model_fold_path = Path(self.model_path) / 'fold_all'
        model_fold_path.mkdir(parents=True, exist_ok=True)
        model = os.path.join(self.model_path, 'fold_all/model.pth')
        
        if not Path(model).exists(): # check if model already exists
            print('\n...Could not find an existing instalation of the model...')
            print('\n...Downloading segmentation model from huggingface...')
            r = requests.get('https://huggingface.co/iantsang779/pyroothair_v1/resolve/main/model.pth')

            with open(model, 'wb') as f:
                f.write(r.content)
            
            print('\n...Model successfully installed...')


    def initialize_model(self, device, override_model_path: str = None, override_model_checkpoint: str = None):
        # https://github.com/MIC-DKFZ/nnUNet/blob/f8f5b494b7226b8b0b1bca34ad3e9b68facb52b8/nnunetv2/inference/predict_from_raw_data.py#L39

        self.predictor = nnUNetPredictor(device=device) # instantiate nnUNet predictor

        if override_model_path or override_model_checkpoint is None: # if using default model 
            # initialize network and load checkpoint
            # self.predictor.initialize_from_trained_model_folder(
            #     join(os.environ.get('nnUNet_results'), 'Dataset999_pyRootHair/nnUNetTrainer__nnUNetResEncUNetMPlans__2d'),
            #     use_folds=('all'),
            #     checkpoint_name='checkpoint_final_no_opt_new.pth')
            self.predictor.initialize_from_trained_model_folder(
                self.model_path,
                use_folds=('all'),
                checkpoint_name='model.pth'
            )

        else: # if user specifies custom model path, load that model instead
            self.predictor.initialize_from_trained_model_folder(
                override_model_path,
                use_folds=(0,),
                checkpoint_name=override_model_checkpoint)
    
    def run_inference(self):
    
        adjusted_img_dir = Path(self.in_dir).parent / 'adjusted_images' / self.run_id # directory containing modified images for nnUNet input
        parent_path = Path(self.in_dir).parent
        mask_dir = parent_path / 'masks'
        mask_dir.mkdir(parents=True, exist_ok=True) # make dir to store masks

        sub_dir = mask_dir / self.run_id
        sub_dir.mkdir(exist_ok=True)

        print(f'\n...Setting up a new directory {Path(sub_dir)} to store the predicted masks ...\n')
        self.predictor.predict_from_files(str(adjusted_img_dir),
                                          str(Path(sub_dir)),
                                          save_probabilities=False,
                                          overwrite=False,
                                          num_processes_preprocessing=2,
                                          num_processes_segmentation_export=2,
                                          num_parts=1,
                                          part_id=0)
        





        
