import torch
import os
import subprocess
import imageio.v3 as iio

from pathlib import Path

class nnUNet():

    def __init__(self):
        self.gpu_exists = False
        self.home_dir = None

    def check_gpu(self) -> None:
        """
        Check whether GPU is available on local machine/cluster
        """
        if torch.cuda.is_available():
            self.gpu_exists = True
            print(f'\n...GPU Detected! Using {torch.cuda.get_device_name(0)} ...\n')
 
    def setup_nnunet_paths(self) -> None:
        """
        Setup nnUnet path for results if it doesn't currently exist. Required for running inference
        """
        if os.environ.get('nnUNet_results') is None: # check whether nnUNet_results exists in local machine
            self.home_dir = Path.home() # get user home directory
            Path(os.path.join(self.home_dir, 'nnUNet_results')).mkdir() # make directory for nnUNet results
            res_path = os.path.join(self.home_dir, 'nnUNet_results')
            os.environ['nnUNet_results'] = res_path # export path to newly created directory (temp)
            print(f'\n...nnUNet paths have been set up...\n')

    def load_model(self, model_path:str) -> None:
        """
        Load pre-trained nnUNet segmentation model from path
        """
        print('\n...Loading nnU-Net model...\n')
        subprocess.run(["nnUNetv2_install_pretrained_model_from_zip", model_path])
        print('\n...Model loaded!...\n')

    def run_inference(self, in_dir:str, out_dir:str):
        """
        Run inference with model on input directory. Outputs predicted masks in output_directory. 
        """
        
        print('\n...Running inference...\n')

        subprocess.run(["nnUNetv2_predict",
                        "-d", "Dataset069_iRootHair",
                        "-i", in_dir,
                        "-o", out_dir,
                        "-f", "0","1","2","3","4", # 5 fold cross-val
                        "-c", "2d", # only trained on 2d config
                        "-tr", "nnUNetTrainer",
                        "-p", "nnUNetResEncUNetLPlans"])
        
        print('\n...Predicted masks have been generated...\n')
    
    
        

