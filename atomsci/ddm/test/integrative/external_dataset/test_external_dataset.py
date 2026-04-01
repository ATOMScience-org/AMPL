#!/usr/bin/env python

import json
import pandas as pd
import os
import sys

import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import predict_from_model as pfm
import pytest

def clean():
    """Clean test files"""
    if "output" not in os.listdir():
        os.mkdir("output")
    for f in os.listdir("./output"):
        if os.path.isfile("./output/"+f):
            os.remove("./output/"+f)

def test():
    """Test AD index calculation: Curate data, fit model, and predict property for new compounds for each feature set"""

    # Clean
    # -----
    clean()

    # Run HyperOpt
    # ------------
    with open("H1_graphconv.json", "r") as f:
        hp_params = json.load(f)

    script_dir = parse.__file__.strip("parameter_parser.py").replace("/pipeline/", "")
    python_path = sys.executable
    hp_params["script_dir"] = script_dir
    hp_params["python_path"] = python_path
    
    params = parse.wrapper(hp_params)
    if not os.path.isfile(params.dataset_key):
        params.dataset_key = os.path.join(params.script_dir, params.dataset_key)

    train_df = pd.read_csv(params.dataset_key)

    pl = mp.ModelPipeline(params)
    pl.train_model()

    # this should raise an exception
    with pytest.raises(Exception, match='Dataset file file_does_not_exist.csv does not exist'):
        pred_df_file = pfm.predict_from_model_file(model_path=pl.params.model_tarball_path,
                                            input_df=train_df[:10],
                                            id_col="compound_id",
                                            smiles_col="base_rdkit_smiles",
                                            response_col="pKi_mean",
                                            dont_standardize=True,
                                            AD_method="z_score",
                                            external_training_data='file_does_not_exist.csv')

    # this should work
    pred_df_file = pfm.predict_from_model_file(model_path=pl.params.model_tarball_path,
                                        input_df=train_df[:10],
                                        id_col="compound_id",
                                        smiles_col="base_rdkit_smiles",
                                        response_col="pKi_mean",
                                        dont_standardize=True,
                                        AD_method="z_score")
    assert("AD_index" in pred_df_file.columns.values), 'Error: No AD_index column in pred_df_file'

    clean()

if __name__ == '__main__':
    test()
