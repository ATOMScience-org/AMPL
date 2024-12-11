#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import sys

import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.utils import llnl_utils


def clean():
    """Clean test files"""
    if "output" not in os.listdir():
        os.mkdir("output")
    for f in os.listdir("./output"):
        if os.path.isfile("./output/"+f):
            os.remove("./output/"+f)

def test():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    clean()

    if not llnl_utils.is_lc_system():
        assert True
        return

    # Run HyperOpt
    # ------------
    with open("H1_hybrid.json", "r") as f:
        hp_params = json.load(f)

    script_dir = parse.__file__.strip("parameter_parser.py").replace("/pipeline/", "")
    python_path = sys.executable
    hp_params["script_dir"] = script_dir
    hp_params["python_path"] = python_path

    params = parse.wrapper(hp_params)
    if not os.path.isfile(params.dataset_key):
        params.dataset_key = os.path.join(params.script_dir, params.dataset_key)

    train_df = pd.read_csv(params.dataset_key)

    print("Train a hybrid models with MOE descriptors")
    pl = mp.ModelPipeline(params)
    pl.train_model()

    print("Check the model performance on validation data")
    pred_data = pl.model_wrapper.get_perf_data(subset="valid", epoch_label="best")
    pred_results = pred_data.get_prediction_results()
    print(pred_results)

    pred_score = pred_results['r2_score']
    score_threshold = 0.4
    assert pred_score > score_threshold, \
        f'Error: Score is too low {pred_score}. Must be higher than {score_threshold}'

    print("Make predictions with the hyrid model")
    predict= pl.predict_on_dataframe(train_df[:10], contains_responses=False)
    assert (predict['pred'].shape[0] == 10), 'Error: Incorrect number of predictions'
    assert (np.all(np.isfinite(predict['pred'].values))), 'Error: Predictions are not numbers'

    
if __name__ == '__main__':
    test()
