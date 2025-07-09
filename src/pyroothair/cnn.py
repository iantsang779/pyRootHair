import torch
import os
import requests
import json

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class nnUNetv2():

    def __init__(self, in_dir:str, run_id:str):
        self.in_dir = in_dir # user input directory containing raw images
        self.run_id = run_id
        self.predictor=None
        self.model_path = os.path.join(Path(__file__).parent, 'model')
        self.gpu_exists=False

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
        """
        Install pyroothair segmentation model + necessary json files from huggingface
        Compare metadata etag to check whether a new model has been pushed to huggingface, get new model/JSONs if true
        """
        model_fold_path = Path(self.model_path) / 'fold_all'
        model_fold_path.mkdir(parents=True, exist_ok=True)

        files = [os.path.join(self.model_path, 'fold_all/model.pth'),
                 os.path.join(self.model_path, 'dataset.json'),
                 os.path.join(self.model_path, 'dataset_fingerprint.json'),
                 os.path.join(self.model_path, 'plans.json')]

        metadata_jsons = [os.path.join(self.model_path, 'model_metadata.json'),
                          os.path.join(self.model_path, 'dataset_metadata.json'),
                          os.path.join(self.model_path, 'dataset_fingerprint_metadata.json'),
                          os.path.join(self.model_path, 'plans_metadata.json')]
        
        url_dict = {'model_url': 'https://huggingface.co/iantsang779/pyroothair_v1/resolve/main/model.pth',
                    'dataset_json_url': 'https://huggingface.co/iantsang779/pyroothair_v1/resolve/main/dataset.json',
                    'dataset_fingerprint_url': 'https://huggingface.co/iantsang779/pyroothair_v1/resolve/main/dataset_fingerprint.json',
                    'plans_url': 'https://huggingface.co/iantsang779/pyroothair_v1/resolve/main/plans.json'}
      
        for url, model_file, metafile in zip(url_dict, files, metadata_jsons):
 
            if not Path(model_file).exists(): # check if file (model/model json) already exists
                download_model = True
                print(f'\n...Could not find an existing local installation of {model_file.split('/')[-1]}...')

            else: # if file already exists
                local_etag = None  

                if Path(metafile).exists(): 
                    with open(metafile, 'r') as f:
                        local_etag = json.load(open(metafile))[url] # get local etag value for each file from metadata.txt
                try:
                    response = requests.get(url_dict[url]) # get remote etag
                    response.raise_for_status()
                    remote_etag = response.headers.get('ETag')

                except requests.RequestException as e:
                    print(f'...Error checking for updates on huggingface: {e}...')
                    remote_etag = None
                
                download_model = remote_etag != local_etag # set download_model to True if remote etag mismatches with local etag

            if download_model:
                print(f'\n...Downloading the latest {model_file.split('/')[-1]} from: {url_dict[url]}...')

                try:
                    r = requests.get(url_dict[url])
                    r.raise_for_status()

                    with open(model_file, 'wb') as f: # download model
                        f.write(r.content)
                    
                    # save metadata to dict
                    field = {url: r.headers.get('Etag')}
                    json.dump(field, open(metafile, 'w')) # update metadata file with most recent metadata
                    
                except requests.RequestException as e:
                    print(f'\n...Download failed: {e}...')
                    if not Path(model_file).exists():
                        raise RuntimeError(f'\n...Download failed and no local installation of {url} can be found...')
                    print(f'\n...Using existing file: {url}...')
        
                print(f'\n...Updated pyRootHair model and JSON files have been successfully installed from Huggingface in: {self.model_path}...')


    def initialize_model(self, device):
        # https://github.com/MIC-DKFZ/nnUNet/blob/f8f5b494b7226b8b0b1bca34ad3e9b68facb52b8/nnunetv2/inference/predict_from_raw_data.py#L39

        self.predictor = nnUNetPredictor(device=device) # instantiate nnUNet predictor
        self.predictor.initialize_from_trained_model_folder(
            self.model_path,
            use_folds=('all'),
            checkpoint_name='model.pth'
            )
    
    def run_inference(self, out_dir:str):
        
        assert self.predictor is not None

        adjusted_img_dir = Path(out_dir) / 'adjusted_images' / self.run_id # directory containing modified images for nnUNet input
        mask_dir = Path(out_dir) / 'masks' / self.run_id
        mask_dir.mkdir(parents=True, exist_ok=True) # make dir to store masks

        print(f'\n...Setting up a new directory {Path(mask_dir)} to store the predicted masks ...\n')
        self.predictor.predict_from_files(str(adjusted_img_dir),
                                          str(Path(mask_dir)),
                                          save_probabilities=False,
                                          overwrite=False,
                                          num_processes_preprocessing=2,
                                          num_processes_segmentation_export=2,
                                          num_parts=1,
                                          part_id=0)
        





        
