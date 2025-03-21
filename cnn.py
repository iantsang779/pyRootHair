import torch
import os

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join


### Load nnUNet via subprocess ###
# class nnUNet():

#     def __init__(self, in_dir:str, run_id:str) -> None:
#         self.gpu_exists = False
#         self.home_dir = None
#         self.in_dir = in_dir # user input directory containing raw images
#         self.run_id = run_id
        

#         print('#########################################')
#         print('     Thank you for using pyRootHair!     ')
#         print('#########################################\n')  

#     def check_gpu(self) -> None:
#         """
#         Check whether GPU is available on local machine/cluster
#         """
        
#         if torch.cuda.is_available():
#             self.gpu_exists = True
#             print(f'\n...GPU Detected! Using {torch.cuda.get_device_name(0)}...\n')
#         else:
#             print(f'\n...No GPU Detected...\n')

#     def setup_nnunet_paths(self) -> None:
#         """
#         Setup nnUnet path for results if it doesn't currently exist. Required for running inference
#         """
#         if os.environ.get('nnUNet_results') is None: # check whether nnUNet_results exists in local machine
#             self.home_dir = Path.home() # get user home directory
#             Path(os.path.join(self.home_dir, 'nnUNet_results')).mkdir() # make directory for nnUNet results
#             res_path = os.path.join(self.home_dir, 'nnUNet_results')
#             os.environ['nnUNet_results'] = res_path # export path to newly created directory (temp)
#             print(f'\n...nnUNet paths have been set up...\n')

#     def load_model(self, model_path:str) -> None:
#         """
#         Load pre-trained nnUNet segmentation model from path
#         """
#         print('\n...Loading nnU-Net model...\n')
#         try:
#             subprocess.run(["nnUNetv2_install_pretrained_model_from_zip", model_path])
    
#             print('\n...Model loaded...\n')

#         except subprocess.CalledProcessError as e:
#             print(f'Failed with error: {e}')

#     def run_inference(self, dataset: str='Dataset069_iRootHair', planner: str='nnUNetResEncUNetLPlans'):
#         """
#         Run inference with model on input directory. Outputs predicted masks in output_directory. 
#         """
        
#         print('\n...Running inference...\n')

#         adjusted_img_dir = Path(self.in_dir).parent / 'adjusted_images' / self.run_id # directory containing modified images for nnUNet input
#         print('Adjusted img dir', adjusted_img_dir)

#         parent_path = Path(self.in_dir).parent
#         mask_dir = parent_path / 'masks'
#         mask_dir.mkdir(parents=True, exist_ok=True) # make dir to store masks

#         sub_dir = mask_dir / self.run_id
#         sub_dir.mkdir(exist_ok=True)

#         print(f'\n...Setting up a new directory {Path(sub_dir)} to store the predicted masks ...\n')


#         try:
#             res = subprocess.run(["nnUNetv2_predict",
#                             "-d", dataset,
#                             "-i", str(adjusted_img_dir),
#                             "-o", str(Path(sub_dir)),
#                             "-f", "0","1","2","3","4", # 5 fold cross-val
#                             "-c", "2d", # only trained on 2d config
#                             "-tr", "nnUNetTrainer",
#                             "-p", planner,
#                             "--disable_progress_bar"],
#                             stdout=subprocess.PIPE,
#                             stderr=subprocess.PIPE,
#                             text=True,
#                             check=True)
#             print(res.stdout)
#             print(res.stderr)
#             print('\n...Inference successful - predicted masks have been generated...\n')

#         except subprocess.CalledProcessError as e:
#             print(f'Failed with error: {e}')
        
### Load nnUNet via python  ###

class nnUNetv2():

    def __init__(self, in_dir:str, run_id:str, args):
        self.in_dir = in_dir # user input directory containing raw images
        self.run_id = run_id
        self.predictor=None
        self.args = args

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

    # def setup_paths(self) -> None:
    #     """
    #     Setup paths for results and model storing.
    #     """
    #     master_pth = Path('~/pyroothair').expanduser() # master directory where everything is stored

    #     if master_pth.is_dir(): # check whether pyroothair directory exists
    #         Path(os.path.join(master_pth, 'nnUNet_results')).mkdir() # make directory for nnUNet results
            
         
            
    #         print(f'\n...nnUNet results path has been set up...\n')
    #     else:
    #         raise ValueError('Missing master folder pyroothair. Please create the folder in your home directory!')
    ### ! Can replace this function by just having the correct directories on github (nnUNet_`results/Dataset..../nnUNet_trainer....)`

    def initialize_model(self, device, override_model_path, override_model_checkpoint):
        # https://github.com/MIC-DKFZ/nnUNet/blob/f8f5b494b7226b8b0b1bca34ad3e9b68facb52b8/nnunetv2/inference/predict_from_raw_data.py#L39

        self.predictor = nnUNetPredictor(device=device) # instantiate nnUNet predictor

        if self.args.override_model_path is None: # if using default model 
            # initialize network and load checkpoint
            self.predictor.initialize_from_trained_model_folder(
                join(os.environ.get('nnUNet_results'), 'Dataset069_iRootHair/nnUNetTrainer__nnUNetResEncUNetLPlans__2d'),
                use_folds=('all'),
                checkpoint_name='checkpoint_final.pth')

        else: # if user specifies custom model path, load that model instead
            self.predictor.initialize_from_trained_model_folder(
                override_model_path,
                use_folds=(0,),
                checkpoint_name=override_model_checkpoint)
    
    def run_inference(self):

        adjusted_img_dir = Path(self.in_dir).parent / 'adjusted_images' / self.run_id # directory containing modified images for nnUNet input
        print('Adjusted img dir', adjusted_img_dir)

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
        





        
