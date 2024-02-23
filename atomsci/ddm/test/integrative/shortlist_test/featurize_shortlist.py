
"""Featurize several files at once"""
import sys
import os
import json
import pandas as pd
 
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
 
def featurize_from_shortlist(shortlist_path=None, split_json = None):
    """Featurize and split the ChEMBL hERG pIC50 dataset. Then create a config
    file for running a hyperparameter search to model this dataset.
    """
    sl = pd.read_csv(shortlist_path)
    with open(split_json, "r") as f:
        hp_params = json.load(f)
    
    print('Featurizing shortlist')
    hp_params.pop('use_shortlist')
    hp_params.pop('shortlist_key')
 
    for i, row in sl.iterrows():
        hp_params['dataset_key'] = row.dataset_key
        hp_params['response_cols'] = row.response_cols
        pparams = parse.wrapper(hp_params)

        print('-----------------------------------------------')
        print(hp_params['dataset_key'])
        print(pparams.dataset_key)
        print('-----------------------------------------------')

        # Create a ModelPipeline object
        pipe = mp.ModelPipeline(pparams)

        # Featurize and split the dataset
        split_uuid = pipe.split_dataset()

        # Delete split file to keep it cleaner
        rdir = hp_params['result_dir']
        dkey = row.dataset_key.replace('.csv','')
        os.remove(f'{dkey}_train_valid_test_scaffold_{split_uuid}.csv')
        
if __name__ == '__main__':
    fp = sys.argv[1]
    split_json = sys.argv[2]
    featurize_from_shortlist(fp, split_json)
    sys.exit(0)